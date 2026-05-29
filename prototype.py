########################################################################
# Interactive Fiction Agentic System
########################################################################

#########################################################################

# SUMMARY:

#########################################################################

import json

from dotenv import load_dotenv
import os
from collections import deque

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
from osbot_utils.type_safe.primitives.domains.identifiers.Edge_Id import Edge_Id
from osbot_utils.type_safe.primitives.domains.identifiers.Node_Id import Node_Id
from typing import Dict


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
    node_data: Custom_Node_Data  # type: ignore
 

# Custom Edge for MGraph
class Custom_Edge_Data(Schema__MGraph__Edge__Data):
    predicate: str = ""  # This allows the 'predicate' key inside edge_data

class Custom_Edge(Schema__MGraph__Edge):
    edge_data: Custom_Edge_Data  # type: ignore


# Custom Graph Schema to preserve Custom_Node and Custom_Edge types when loading
class Custom_Graph(Schema__MGraph__Graph):
    nodes: Dict[Node_Id, Custom_Node]  # type: ignore
    edges: Dict[Edge_Id, Custom_Edge]  # type: ignore
    

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
    predicate: str = Field(description="The predicate of the relationship")
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
        You are an expert story teller
        You will be given tools to query a knowledge graph that contains information about a plot, including entities and relationships between them.
        You will perform the following tasks based on the users input:
        1. Continue the story based on the information in the knowledge graph, ensuring that the story remains coherent and consistent with the plot as represented in the graph.
        2. Answer any questions the user has about the plot based on the information in the knowledge graph.
        3. Provide a summary of chnages to the story based on the players actions, which will be reflected as updates to the knowledge graph.
                   
        Story should always be written from the characters perspective, and should be engaging and immersive, drawing the user into the world of the story and allowing them to experience it from the character's point of view.
    """)
)

story_updater_agent = Agent(
    model,
    output_type=str,
    system_prompt=("""
        You are an expert story updater.
        You will be given a user's input and a memory of the story so far, which includes previous user inputs and the corresponding story updates that were generated in response to those inputs.
        Your task is to generate updates the knowledge graph representing the changes to the story based on the user's input and the memory of the story so far.
        Once completed you will respond with a summary of the changes that were made to the knowledge graph.
    """)
)

character_agent = Agent(
    model,
    output_type=str,
    system_prompt=("""
        You are an expert at understanding and summarizing a story from a specific character's perspective.
        You will be given the relevant triplets from a knowledge graph that contains information from the Characters persepctive.
        You will be given the history of the users actions actions and corresponding story updates triggered by those actions.
        You will be given tools to query the knowledge graph, add relationships, and remove relationships.
        Your task is
            * Generate the most likely action this character would take based on the information available.
            * Update the knowledge graph based on the character's actions as needed.
    """)
)

guardian_agent = Agent(
    model,
    output_type=str,
    system_prompt=("""
        You are a guardian agent, responsible for making sure that inputs and outputs from the story teller agent are appropriate and do not contain any harmful or inappropriate content.
        You will review the story updates generated by the story teller agent and ensure that they adhere to ethical guidelines and do not contain any content that could be considered harmful, offensive, or inappropriate.
        If you find any content that violates these guidelines, you will flag it and prevent it from being presented to the user, and respond with "CANNOT UPDATE".

        Inappropriate Content includes, but is not limited to:
        * Hate speech or discriminatory language
        * Explicit or graphic content that is not suitable for all audiences
        * Content that promotes violence or self-harm
        * Any content that could be considered offensive or harmful to individuals or groups based on factors such as race, religion, or sexual orientation.
        * Any content that violates the terms of service of the platform on which this system is being used.
        * Any content that violates legal or ethical standards for content in the relevant jurisdiction.                      
   """)
)

########################################################################
# MGraph Query and Update Tools
########################################################################

@story_teller_agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
def query_kg_by_entity_name(entity_name: str) -> str:
    """
    Queries the knowledge graph for a specific entity and its relationships.

    Args:
        entity_name: The name of the entity to query for.
    Returns:
        A summary of the specified entity and its relationships in the knowledge graph.
    """
    results = []

    target_node_id = None

    with mgraph.data() as data:
        for node in data.nodes():
            if hasattr(node, 'node_data') and getattr(node.node_data, 'name', None) == entity_name:
                target_node_id = node.node_id
                break    
    
        for edge in data.edges():
            if edge.from_node_id() == target_node_id or edge.to_node_id() == target_node_id:
        
                subject = data.node(edge.from_node_id()).node_data.name
                predicate = getattr(edge.edge.data.edge_data, 'predicate', None)
                object = data.node(edge.to_node_id()).node_data.name
        
                results.append((subject, predicate, object))

    return str(results)


@story_updater_agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
@character_agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
def insert_triplet_into_kg(subject: str, predicate: str, object: str):
    """
    Inserts a new triplet into the knowledge graph, creating nodes for the subject and object if they do not already exist, and an edge for the predicate that connects them.

    Args:
        subject: The subject of the relationship to insert into the knowledge graph.
        predicate: The predicate of the relationship to insert into the knowledge graph.
        object: The object of the relationship to insert into the knowledge graph.
    """
    subject_node_id = None
    object_node_id = None

    with mgraph.data() as data:
        for node in data.nodes():
            if hasattr(node, 'node_data'):
                if getattr(node.node_data, 'name', None) == subject:
                    subject_node_id = node.node_id
                elif getattr(node.node_data, 'name', None) == object:
                    object_node_id = node.node_id
    
    with mgraph.edit() as edit:
        if subject_node_id is None:
            subject_node = edit.new_node(
                node_type=Custom_Node, # type: ignore
                name=subject,
                type="Unknown",
                description=""
            )
            subject_node_id = subject_node.node_id

        if object_node_id is None:
            object_node = edit.new_node(
                node_type=Custom_Node, # type: ignore
                name=object,
                type="Unknown",
                description=""
            )
            object_node_id = object_node.node_id

        edit.new_edge(
            edge_type=Custom_Edge,      # Tells mgraph to use your new schema
            from_node_id=subject_node_id,
            to_node_id=object_node_id,
            edge_data={
                "predicate": predicate  # This will now bypass the type-checker!
            }
        )

@story_updater_agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
@character_agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
def remove_relationship(subject: str, predicate: str, object: str):
    """
    Removes a triplet from the knowledge graph based on the subject, predicate, and object.
    NOTE: Only the predicate is removed, the nodes for the subject and object will remain in the graph. This is because it can be difficult to determine if a node should be deleted based on a single relationship, as it may have other relationships that should be preserved. In a more complex implementation, you could add additional logic to check if the nodes have any other relationships before deciding to delete them.

    Args:
        subject: The subject of the relationship to remove from the knowledge graph.
        predicate: The predicate of the relationship to remove from the knowledge graph.
        object: The object of the relationship to remove from the knowledge graph.
    """
    with mgraph.edit() as edit:
        for edge in edit.edges():  # type: ignore
            edge_subject = edit.node(edge.from_node_id()).node_data.name  # type: ignore
            edge_predicate = getattr(edge.edge_data, 'predicate', None)
            edge_object = edit.node(edge.to_node_id()).node_data.name  # type: ignore

            if edge_subject == subject and edge_predicate == predicate and edge_object == object:
                edit.delete_edge(edge.edge_id)

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
        characters=character_entities_results.output, # type: ignore
        locations=location_entities_results.output,  # type: ignore
        objects=object_entities_results.output,  # type: ignore
        plot_events=plot_event_entities_results.output,  # type: ignore
        themes=theme_entities_results.output,  # type: ignore
        genre=genre_entities_results.output  # type: ignore
    ) 


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
    parser.add_argument('--graph_file', type=str, default='knowledge_graph.json', help='Path to the knowledge graph file (if loading an existing graph)')
    parser.add_argument('--memory_length', type=int, default=5, help='The number of previous interactions to keep in memory for context during the interactive phase')

    return parser.parse_args()


def graph_construction_pipeline(plot_file: str):
    """
    Constructs a knowledge graph from the given plot file by extracting entities and relationships.
    The KG only needs to be constructed once per plot, and can be updated as needed during the interaction phase.

    Args:
        plot_file: The path to the plot file from which to construct the knowledge graph.
    """
    plot = load_plot_file(filename=plot_file)

    guardrails_check = guardian_agent.run_sync(f"Review the following plot and determine if it contains any content that would be considered inappropriate based on the guidelines you follow. If it does, respond with 'CANNOT CONSTRUCT GRAPH'. If it does not, respond with 'GRAPH CONSTRUCTION APPROVED'. Plot: {plot}").output # type: ignore
    if "CANNOT UPDATE" in guardrails_check:
        print("This plot contains content that violates the guidelines. Cannot construct knowledge graph.")
        return False


    entity_collection = load_entities_from_plot(plot)
    relationships = extract_relationships(plot, entity_collection.characters + entity_collection.locations + entity_collection.objects + entity_collection.plot_events + entity_collection.themes + entity_collection.genre)
    construct_knowledge_graph(relationships.output)  # type: ignore
    return True


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

    graph_schema = Custom_Graph.from_json(data)
    mgraph.graph.model.data = graph_schema  # type: ignore
    mgraph.edit().rebuild_index()


def get_all_characters() -> list[str]:
    """
    Retrieves a list of all character names from the knowledge graph. This can be used to provide options to the user for selecting a character's perspective to experience the story from.
    """
    characters = []
    with mgraph.data() as data:
        for node in data.nodes():
            if hasattr(node, 'node_data') and getattr(node.node_data, 'type', None) == "Character":
                characters.append(getattr(node.node_data, 'name', None))
    return characters

def get_story_so_far(character: str) -> str:
    """
    Retrieves the "story so far" based on the current state of the knowledge graph. This can be used to provide context to the user during interactions, 
    allowing them to see a summary of the plot and the relationships between entities as they explore the story.
    """

    character_triplets = query_kg_by_entity_name(character)

    prompt = f""" 
    You are an expert story teller and summarizer.
    You will be given the relevant triplets from a knowledge graph that contains information from the Characters persepctive.
    You will be given a character from whoms perspective the interactive story will be experienced.
    Your task is to:

    * Summarize the story so far based on the information in the knowledge graph, from the perspective of the given character.
    * If you cannot find information on the character in the graph, you should respond with "I don't know much about {character} yet."

    Character: 
    {character}
    Relevant Triplets from the Knowledge Graph:
    {character_triplets}
    """

    return story_teller_agent.run_sync(prompt).output # type: ignore

def generate_story_update(user_input: str, memory: list) -> str:
    """
    Generates a story update based on the user's input and the current state of the knowledge graph. This function can be used to create dynamic story updates that reflect the user's actions and choices, allowing for an interactive storytelling experience.
    """

    prompt = f"""
    You are an expert story teller and summarizer.
    You will be given tools to query a knowledge graph that contains information about a plot, including entities and relationships between them.
    You will be given a list of the users previous inputs and the corresponding story updates that were generated in response to those inputs, which will serve as the memory of the story so far.
    You will perform the following tasks based on the users input:
    1. Summarize the story so far based on the information in the knowledge graph.
    2. Answer any questions the user has about the plot based on the information in the knowledge graph.
    3. Provide a summary of chnages to the story based on the players actions, which will be reflected as updates to the knowledge graph.

    User Input:
    {user_input}

    Memory of the story so far (previous user inputs and corresponding story updates):
    {memory}
    """

    return story_teller_agent.run_sync(prompt).output # type: ignore

def update_story_and_kg(user_input: str, story_update: str, memory: list) -> str:
    """
    Updates the story and knowledge graph based on the user's input and the generated story update. This function can be used to reflect changes to the story in the knowledge graph, allowing for a dynamic and evolving storytelling experience.
    """

    prompt = f"""
        Update the knowledge graph based on the following user input and story update, which reflects changes to the story. The story update is based on the user's input and the current state of the knowledge graph, and may include new entities, relationships, or changes to existing entities and relationships in the graph.

        user_input: {user_input}
        story_update: {story_update}
        memory: {memory}
    """

    return story_updater_agent.run_sync(prompt).output # type: ignore


def character_agent_interaction(character: str, user_input: str, memory: list) -> str:
    """
    Interacts with the character agent to determine the character's next actions in the story based on the user's input and the current state of the knowledge graph, and updates the knowledge graph based on the character's actions.

    Args:
        character: The name of the character whose perspective the story is being told from.
        user_input: The user's input that may influence the character's actions.
        memory: A list of previous user inputs and corresponding story updates that provide context for the character's decision-making.
    Returns:
        A summary of the character's next actions in the story based on the user's input and the current state of the knowledge graph.
    """

    character_triplets = query_kg_by_entity_name(character)

    prompt = f"""
        Update the actions and provide the perpective for the character based on the following user input and memory of the story so far, which includes previous user inputs and corresponding story updates. The character's actions should be based on the information in the knowledge graph, which includes relevant triplets about the character and their relationships to other entities in the graph.
            
        Character: 
        {character}

        Relevant Triplets from the Knowledge Graph:
        {character_triplets}

        User Input:
        {user_input}

        Memory of previous user inputs and corresponding story updates:
        {memory}
    """

    return character_agent.run_sync(prompt).output # type: ignore

########################################################################
# Main Entrypoint
########################################################################

def main():
    command_line_args = get_input_args()

    # Phase 1: Parse the Plot into a KG.
    if command_line_args.construct_graph:
        graph_ok = graph_construction_pipeline(command_line_args.plot_file)
        if not graph_ok:
            return
    else:
        load_knowledge_graph(command_line_args.graph_file)    

    # Phase 2: Interact with the user using the KG.
    character_name = None

    characters = get_all_characters()
    
    print("Available characters to experience the story from:")
    for i, character in enumerate(characters):
        print(f"{i + 1}. {character}")
    
    while character_name is None:
        character_choice = input("Please select a character by entering the corresponding number: ")
        try:
            character_index = int(character_choice) - 1
            if 0 <= character_index < len(characters):
                character_name = characters[character_index]
            else:
                print("Invalid choice. Please enter a number corresponding to one of the characters listed.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    characters.remove(character_name) # remove the selected character from the list of characters, so that we can use that list for updating the story from the other characters perspectives during interactions
    

    print(f"Getting the story so far from the perspective of {character_name}...")
    story_so_far = get_story_so_far(character_name)
    print(f"Story so far from {character_name}'s perspective:\n{story_so_far}")

    session_memory = deque(maxlen=command_line_args.memory_length)

    while True:
        user_input = input("What would you like to do? (Type 'exit' or 'quit' to quit, 'memory' to see memory, 'export' to export the current knowledge graph) ")
        if user_input.lower() == 'exit' or user_input.lower() == 'quit':
            break
        
        if user_input.lower() == 'memory':
            print("Memory of the story so far (previous user inputs and corresponding story updates):")
            for entry in session_memory:
                print(f"User Input: {entry['user_input']}")
                print(f"Story Update: {entry['story_update']}")
                print(f"KG Changes: {entry['kg_changes']}")
                print("-----")
            continue

        if user_input.lower() == 'export':
            export_file = input("Enter the filename to export the knowledge graph to (e.g., 'updated_knowledge_graph.json'): ")

            if export_file.strip() == "":
                export_file = "updated_knowledge_graph.json"

            print(f"Exporting knowledge graph to {export_file}...")
            with mgraph.export() as export:
                data = export.to__mgraph_json()
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            continue

        guardian_response = guardian_agent.run_sync(user_input).output # type: ignore
        if "CANNOT UPDATE" in guardian_response:
            print("Sorry, I cannot process that input. Please try something else.")
            continue

        print("Generating story update based on user input and memory of the story so far...")
        story_update = generate_story_update(user_input, list(session_memory))

        print("Updating the story and knowledge graph based on the generated story update...")
        kg_changes = update_story_and_kg(user_input, story_update, list(session_memory))

        memory_entry = {
            "user_input": user_input,
            "story_update": story_update,
            "kg_changes": kg_changes
        }
        session_memory.append(memory_entry)


        ########### CHARACTER AND WORLD UPDATES ############
        for character in characters:
            print(f"Updating story from {character}'s perspective...")
            update = character_agent_interaction(character, user_input, list(session_memory))
            print(f"Story update from {character}'s perspective:\n{update}")


        ######## PRINT STORY UPDATE FOR USER #########
        print(f"*************************************************************\n{story_update}")


if __name__ == "__main__":
    main()