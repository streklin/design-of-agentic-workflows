########################################################################
# Interactive Fiction Agentic System
########################################################################
import json

from dotenv import load_dotenv
import os

import argparse

from pydantic_ai import Agent
from pydantic import BaseModel, Field

from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.anthropic import AnthropicModel

from mgraph_db.mgraph.MGraph import MGraph
from mgraph_db.mgraph.schemas.Schema__MGraph__Node import Schema__MGraph__Node
from mgraph_db.mgraph.schemas.Schema__MGraph__Node__Data import Schema__MGraph__Node__Data
from mgraph_db.mgraph.schemas.Schema__MGraph__Edge import Schema__MGraph__Edge, Schema__MGraph__Edge__Data
from mgraph_db.mgraph.schemas.Schema__MGraph__Graph import Schema__MGraph__Graph


# Load environment variables from .env file
load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')


########################################################################
# MGraph Objects
########################################################################

# Initialize the in-memory graph structure for storing the knowledge graph.
mgraph = MGraph()

# Custom Node for MGraph
class Custom_Node_Data(Schema__MGraph__Node__Data):
    name: str
    type: str
    description: str
    
class Custom_Node(Schema__MGraph__Node):
    node_data: Custom_Node_Data


# Custom Edge for MGraph
class Custom_Edge_Data(Schema__MGraph__Edge__Data):
    predicate: str = ""  # This allows the 'predicate' key inside edge_data

class Custom_Edge(Schema__MGraph__Edge):
    edge_data: Custom_Edge_Data

########################################################################
# Pydantic Objects
########################################################################

class PlotEntity(BaseModel):
    name: str = Field(description="Name of the entity extracted from the plot")
    type: str = Field(description="The type of entity")
    description: str = Field(description="Description of the extracted entity")

class EntityCollection(BaseModel):
    characters: list[PlotEntity] = Field(description="List of character entities extracted from the plot")
    locations: list[PlotEntity] = Field(description="List of location entities extracted from the plot")
    objects: list[PlotEntity] = Field(description="List of object entities extracted from the plot")
    plot_events: list[PlotEntity] = Field(description="List of plot event entities extracted from the plot")
    themes: list[PlotEntity] = Field(description="List of theme entities extracted from the plot")
    genre: list[PlotEntity] = Field(description="List of genre entities extracted from the plot")

class GraphTriplet(BaseModel):
    subject: PlotEntity = Field(description="The subject of the relationship")
    predicate: str = Field(description="The predicate describing the relationship")
    object: PlotEntity = Field(description="The object of the relationship")


########################################################################
#AGENTS
########################################################################

model = AnthropicModel(
     'claude-haiku-4-5', provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY)
)

entity_extraction_agent = Agent(
    model,
    output_type=list[PlotEntity],
    system_prompt=("""You are an expert at extracting entities from a plot. You will be given a plot and a competency question. 
                   Your task is to extract the relevant entities from the plot that answer the competency question. 
                   Please provide your answer in a structured format, listing each entity with its name, type, and a brief description.""")
) 

triplet_extraction_agent = Agent(
    model,
    output_type=list[GraphTriplet],
    system_prompt=("""You are an expert at extracting relationships between entities in a plot. You will be given a plot and a list of entities extracted from the plot. 
                   Your task is to identify the relationships between these entities based on the information provided in the plot. 
                   Please provide your answer in a structured format, listing each relationship as a triplet with the subject, predicate, and object.""")
)

story_teller_agent = Agent(
    model,
    output_type=str,
    system_prompt=("""
        You are an expert story teller and summarizer.
        You will be given tools to query a knowledge graph that contains information about a plot, including entities and relationships between them.
        You will perform the following tasks based on the users input:
        1. Summarize the story so far based on the information in the knowledge graph.
        2. Answer any questions the user has about the plot based on the information in the knowledge graph.
        3. Provide suggestions for next steps in the story based on the information in the knowledge graph and the user's input.
        4. Update the story based on the user's input and any new information that may be added to the knowledge graph as a result of the user's input.
    """)
)

########################################################################
# MGraph Query and Update Tools
########################################################################

# query entity and relationships

# add a new triplet to the graph

# update an existing triplet in the graph

# remove a triplet from the graph


########################################################################
# Startup/DB Generation Phase
########################################################################

def load_plot_file(filename:str="plot.md"):
    """
    loads the plot file from which the initial KG will be constructed.

    Args:
        filename: filename for the plot file.
    Returns:
        contents of the plot file.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        return content


def extract_entities(plot: str, cq: str, entity_type: str) -> list[PlotEntity]:
    """
    Extracts entities from the plot based on the given competency question.

    Args:
        plot: The plot from which to extract entities.
        cq: The competency question guiding the extraction process.
        entity_type: The type of entities to extract.
    """
    prompt = f"""
                Give each extracted Entity, the type: {entity_type}.

                Competency Question:
                {cq}
                
                Plot:
                {plot}"""
    
    print(f"Extracting entities for CQ: {cq}")
    return entity_extraction_agent.run_sync(prompt) # type: ignore


def load_entities_from_plot(plot: str) -> EntityCollection:
    """
    Loads the plot file and extracts entities based on predefined competency questions.
    
    Args:
        plot: text of the plot
    Returns:
        An EntityCollection containing the extracted entities.
    """

    character_extraction_cq = "What are the main characters in the plot and their roles?"
    location_extraction_cq = "What are the main locations in the plot and their significance?"
    object_extraction_cq = "What are the key objects in the plot and their importance?" 
    plot_event_extraction_cq = "What are the major events in the plot and their impact on the story?"
    theme_extraction_cq = "What are the central themes in the plot and how are they represented?"
    genre_extraction_cq = "What is the genre of the plot and what are its defining characteristics?"

    
    character_entities_results = extract_entities(plot, character_extraction_cq, "Character")
    location_entities_results = extract_entities(plot, location_extraction_cq, "Location")
    object_entities_results = extract_entities(plot, object_extraction_cq, "Object")
    plot_event_entities_results = extract_entities(plot, plot_event_extraction_cq, "PlotEvent")
    theme_entities_results = extract_entities(plot, theme_extraction_cq, "Theme")
    genre_entities_results = extract_entities(plot, genre_extraction_cq, "Genre")

    return EntityCollection(
        characters=character_entities_results.output,
        locations=location_entities_results.output,
        objects=object_entities_results.output,
        plot_events=plot_event_entities_results.output,
        themes=theme_entities_results.output,
        genre=genre_entities_results.output
    )  # type: ignore


def extract_relationships(plot: str, entities: list[PlotEntity]) -> list[GraphTriplet]:
    """
    Creates the basis of a Knowledge Graph by extracting relationships between entities based on the plot.

    Args:
        plot: The plot from which to extract relationships.
        entities: The list of entities extracted from the plot to consider when identifying relationships.
    Returns:        
        list of GraphTriplet representing the relationships between entities in the plot.
    """
    prompt = f"""
    Given the following plot and list of entities, identify the relationships between the entities based on the information provided in the plot.

    Plot:
    {plot}

    Entities:
    {entities}
    """
    print("Extracting relationships between entities...")
    return triplet_extraction_agent.run_sync(prompt) # type: ignore



def construct_knowledge_graph(relationships: list[GraphTriplet], output_file: str = "knowledge_graph.json"):
    """
    Constructs a knowledge graph from the extracted entities and relationships. This function can be implemented using a graph database or an in-memory graph structure depending on the requirements of the system.
    Clears out the existing KG before initiating construction.
    
    Args:
        relationships: The list of relationships between entities extracted from the plot.
    """

    # We build the KG from unique Entity names, so we need to keep track of which entities we've already added to the graph to avoid duplicates.
    # We can use a dictionary to map entity names to their corresponding node IDs in the graph, which will allow us to easily reference existing nodes when adding relationships.
    entities = {}

    with mgraph.edit() as edit:
        for triplet in relationships:
            
            subject_id = None
            object_id = None

            # insert base entites as nodes in the graph if they haven't already been added, and keep track of their node IDs in the entities dictionary
            if triplet.subject.name in entities:
                subject_id = entities[triplet.subject.name]
            else:

                subject = edit.new_node(
                    node_type=Custom_Node, # type: ignore
                    name=triplet.subject.name,
                    type=triplet.subject.type,
                    description=triplet.subject.description
                )

                subject_id = subject.node_id
                entities[triplet.subject.name] = subject_id

            if triplet.object.name in entities:
                object_id = entities[triplet.object.name]
            else:
                object = edit.new_node(
                    node_type=Custom_Node, # type: ignore
                    name=triplet.object.name,
                    type=triplet.object.type,
                    description=triplet.object.description
                )
                object_id = object.node_id
                entities[triplet.object.name] = object_id
            
            # insert relationship as an edge in the graph, referencing the node IDs of the subject and object
            edit.new_edge(
                edge_type=Custom_Edge,      # Tells mgraph to use your new schema
                from_node_id=subject_id,
                to_node_id=object_id,
                edge_data={
                    "predicate": triplet.predicate  # This will now bypass the type-checker!
                }
            )

    # save the graph
    print(f"Saving knowledge graph to {output_file}...")
    with mgraph.export() as export:
        data = export.to__mgraph_json()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def get_input_args():
    """
    Retrieves and parses the command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_file', type=str, default='plot.md', help='Path to the plot file')
    parser.add_argument('--construct_graph', action='store_true', help='Whether to construct the knowledge graph')
    parser.add_argument('--load_graph', action='store_true', help='Whether to load an existing knowledge graph')
    parser.add_argument('--graph_file', type=str, default='knowledge_graph.json', help='Path to the knowledge graph file (if loading an existing graph)')

    return parser.parse_args()


def graph_construction_pipeline(plot_file: str):
    """
    Constructs a knowledge graph from the given plot file by extracting entities and relationships.
    The KG only needs to be constructed once per plot, and can be updated as needed during the interaction phase.

    Args:
        plot_file: The path to the plot file from which to construct the knowledge graph.
    """
    plot = load_plot_file(filename=plot_file)
    entity_collection = load_entities_from_plot(plot)
    relationships = extract_relationships(plot, entity_collection.characters + entity_collection.locations + entity_collection.objects + entity_collection.plot_events + entity_collection.themes + entity_collection.genre)
    construct_knowledge_graph(relationships.output)


########################################################################
# Interactive/RAG Phase
########################################################################

def load_knowledge_graph(graph_file: str):
    """
    Loads an existing knowledge graph from a file. This can be used to load a previously constructed KG without needing to reconstruct it from the plot.

    Args:
        graph_file: The path to the knowledge graph file to load.
    """
    print(f"Loading knowledge graph from {graph_file}...")
    with open(graph_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    graph_schema = Schema__MGraph__Graph.from_json(data)
    mgraph.graph.model.data = graph_schema
    mgraph.edit().rebuild_index()


def get_story_so_far():
    """
    Retrieves the "story so far" based on the current state of the knowledge graph. This can be used to provide context to the user during interactions, 
    allowing them to see a summary of the plot and the relationships between entities as they explore the story.
    """
    pass
    

########################################################################
# Main Entrypoint
########################################################################

def main():
    command_line_args = get_input_args()

    # Phase 1: Parse the Plot into a KG.
    if command_line_args.construct_graph:
        graph_construction_pipeline(command_line_args.plot_file)
    else:
        load_knowledge_graph(command_line_args.graph_file)    

    # Phase 2: Interact with the user using the KG.
    


if __name__ == "__main__":
    main()