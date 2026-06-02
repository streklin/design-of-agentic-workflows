########################################################################
# Interactive Fiction Agentic System
########################################################################

########################################################################
# GOAL:
# The goal of this agentic workflow is to create an interactive fictional 
# environment powered by Large Language Models. The idea is to take a 
# sample story and use it as a basis for various agents to simulate how
# characters in that story would react to a players actions.
########################################################################

########################################################################
# Architecture and System Scope
# This is a multi-agent system. 
# There are two sets of agents in this system, each working together
# to solve a different part of the problem.
# 
# Part 1: Populates a Graph DB based on a story file.
#
# Agents: 
#   Character Extraction Agent: Extracts all key characters from the plot file.
#
#   Location Extraction Agent: Extracts all key locations from the plot file.
#
#   Map Creation Agent: Uses the Locations and the Story to generate triplets
#   that represent an abstract representation of how locations are connected in 
#   the story. These triplets are then used to populate the graphdb.
#
#   Character Relationship Agent: Uses the list of Characters and the plot file
#   to generate triplets that represents how characters and objects in the story
#   are associated with each other. These triplets are then used to populate
#   the graphdb.
#
# Part 2: Use RAG to answer questions about the story and model the
# relationships / locations.
#
# Agents:
#   State Summerization Agent: This agent summerizes the current state of the
#   story based on the contents of the knowledge graph.
# 
#   Story Teller Agent: This Agent specializes in taking in all the actions
#   of the avatar agents and compiling them into a coherent update to the story.
# 
#   Avatar Agent: Agents of this type specialize in acting on behalf of a
#   a character from the story. They generate actions and update the knowledge
#   graph on behalf of the character.
# 
#   Informational Agent: This agent specializes in answering direct queries about
#   characters or relationships in knowledge graph.
# 
#   Routing Agent: This agent specializes in determining if a users input is a
#   question about the story, or an attempt to add to the story.
#
#   Guardian Agent: This agent checks the users input to ensure that meets
#   ethical and safety standards.
########################################################################


import json

from dotenv import load_dotenv
import os
from collections import deque

import argparse

from pydantic import BaseModel, Field


from pydantic_ai import Agent
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
from typing import List

import asyncio


# Load environment variables from .env file
load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

########################################################################
# Pydantic Objects
########################################################################

class PlotEntity(BaseModel):
    name: str = Field(description="Name of the entity extracted from the plot")
    description: str = Field(description="Description of the extracted entity")
    type: str = Field(description="Type of the entity (e.g., character, location, object, event, theme, genre)")

class EntityCollection(BaseModel):
    characters: List[PlotEntity] = Field(description="List of character entities extracted from the plot")
    locations: List[PlotEntity] = Field(description="List of location entities extracted from the plot")
    objects: List[PlotEntity] = Field(description="List of object entities extracted from the plot")
    plot_events: List[PlotEntity] = Field(description="List of plot event entities extracted from the plot")
    themes: List[PlotEntity] = Field(description="List of theme entities extracted from the plot")
    genre: List[PlotEntity] = Field(description="List of genre entities extracted from the plot")

class GraphTriplet(BaseModel):
    subject: PlotEntity = Field(description="The subject of the relationship")
    predicate: str = Field(description="The predicate of the relationship")
    object: PlotEntity = Field(description="The object of the relationship")



########################################################################
# MGraph Objects
########################################################################


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


class MGraphManager:
    """Manager class for handling MGraph operations."""

    def __init__(self):
        self.mgraph = MGraph()
    
    def restore_graph(self, graph_file="knowledge_graph.json"):
        """Restores the MGraph from a JSON file, ensuring that the custom node and edge types are preserved."""
        print(f"Loading knowledge graph from {graph_file}...")
        with open(graph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph_schema = Custom_Graph.from_json(data)
        self.mgraph.graph.model.data = graph_schema  # type: ignore
        self.mgraph.edit().rebuild_index()

    def insert_triplet_list(self, relationships: list[GraphTriplet]):
        """Inserts a list of triplets into the MGraph."""
        
        # We build the KG from unique Entity names, so we need to keep track of which entities we've already added to the graph to avoid duplicates.
        # We can use a dictionary to map entity names to their corresponding node IDs in the graph, which will allow us to easily reference existing nodes when adding relationships.
        
        print(f"Inserting {len(relationships)} triplets into graph...")
        entities = {}

        with self.mgraph.edit() as edit:
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

    def export_graph(self, output_file="exported_graph.json"):
        """Exports the current state of the MGraph to a JSON file."""
        with self.mgraph.export() as export:
            data = export.to__mgraph_json()
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    def query_by_entity_type(self, entity_type: str):
        """
        Queries the knowledge graph for all entities of a specific type and their relationships.

        Args:
            entity_type: The type of entities to query for (e.g., character, location, object, event, theme, genre).
        Returns:
            A summary of all entities of the specified type and their relationships in the knowledge graph.
        """
        print(f"Querying graph for entities of type: {entity_type}")
        results = []

        with self.mgraph.data() as data:
            for node in data.nodes():
                if hasattr(node, 'node_data') and getattr(node.node_data, 'type', None) == entity_type:
                    results.append(getattr(node.node_data, 'name', None))

        return results

    def query_by_entity_name(self, entity_name: str):
        """
        Queries the knowledge graph for a specific entity and its relationships.

        Args:
            entity_name: The name of the entity to query for.

        Returns:
            A summary of the specified entity and its relationships in the knowledge graph.
        """
        print(f"Querying graph for entities of name: {entity_name}")

        results = []

        target_node_id = None

        with self.mgraph.data() as data:
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
    

    def insert_predicate(self, subject: str, predicate: str, object: str):
        """
        Inserts a new triplet into the knowledge graph, creating nodes for the subject and object if they do not already exist, and an edge for the predicate that connects them.

        Args:
            subject: The subject of the relationship to insert into the knowledge graph.
            predicate: The predicate of the relationship to insert into the knowledge graph.
            object: The object of the relationship to insert into the knowledge graph.

        Returns:
            True if task completed, False if an error ocurred.
        """

        print(f"Inserting triplet {subject}->{predicate}->{object}")

        try:
            subject_node_id = None
            object_node_id = None

            with self.mgraph.data() as data:
                for node in data.nodes():
                    if hasattr(node, 'node_data'):
                        if getattr(node.node_data, 'name', None) == subject:
                            subject_node_id = node.node_id
                        elif getattr(node.node_data, 'name', None) == object:
                            object_node_id = node.node_id
            
            with self.mgraph.edit() as edit:
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
                return True
        except:
            return False

    def remove_predicate(self, subject: str, predicate: str, object: str):
        """
        Removes a specific predicate (relationship) between two entities in the knowledge graph.

        Args:
            subject: The subject of the relationship to remove from the knowledge graph.
            predicate: The predicate of the relationship to remove from the knowledge graph.
            object: The object of the relationship to remove from the knowledge graph.
        """
        print(f"Removing triplet {subject}->{predicate}->{object}")


        try:

            with self.mgraph.edit() as edit:
                for edge in edit.edges():  # type: ignore
                    edge_subject = edit.node(edge.from_node_id()).node_data.name  # type: ignore
                    edge_predicate = getattr(edge.edge_data, 'predicate', None)
                    edge_object = edit.node(edge.to_node_id()).node_data.name  # type: ignore

                    if edge_subject == subject and edge_predicate == predicate and edge_object == object:
                        edit.delete_edge(edge.edge_id)
            return True
        except:
            return False

########################################################################
# AGENTS
########################################################################

model = AnthropicModel(
     'claude-haiku-4-5', provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY)
)

class WorldGenerationSystem:

    def _load_plotfile(self, plot_file="plot.md"):
        """
        loads the plot file from which the initial KG will be constructed.

        Args:
            filename: filename for the plot file.
        Returns:
            contents of the plot file.
        """
        with open(plot_file, 'r', encoding='utf-8') as file:
            content = file.read()
            return content

    def __init__(self, graphManager: MGraphManager):
        
        self.graphManager = graphManager if graphManager else MGraphManager()

        self.character_extraction_agent = Agent(
            model,
            output_type=List[PlotEntity],
            system_prompt=f"""
            You are an expert in identifying characters in a story.
            You will be given a story.
            Your task is to:
            * identify all primary characters in the story.
            * generate a description of the character from story.
            """
        )

        self.location_extraction_agent = Agent(
            model,
            output_type=List[PlotEntity],
            system_prompt=f"""
            You are an expert in identifying locations in a story.
            You will be given a story.
            Your task is to:
            * identify all primary locations in the story.
            * generate a description of the location based on the story.
            """
        )

        self.map_creation_agent = Agent(
            model,
            output_type=List[GraphTriplet],
            system_prompt=f"""
            You are an expert at mapping out the connections between locations from a story.
            You will be given a story.
            You will be given a list of locations from that story.
            Your task is to:
            * Construct a collection of triplets where each triplet represents a connection between two locations in the story.
            * Assign the "connects_to" label as the predicate for each triplet.
            """
        )

        self.character_relationship_agent = Agent(
            model,
            output_type=List[GraphTriplet],
            system_prompt=f"""
            You are an expert in mapping out the relationships between characters in a story.
            You are also an expert in mapping out the relationships between characters and objects in a story.
            You will be given a story.
            You will be given a list of characters and objects from that story.
            You will be given a list of locations from that story.
            Your task is to:
            * construct a collection of subject, predicate, object triplets representing the important relationships in the story.
            """
        )


    def _construct_initial_kg(self, location_triplets, character_relationship_triplets):
        """Constructs the initial KG from the extracted triplets."""
        self.graphManager.insert_triplet_list(location_triplets)
        self.graphManager.insert_triplet_list(character_relationship_triplets)
        

    async def process_plot(self, plot_file="plot.md"):
        """
        Processes a plot into a KG.

        Args:
            plot_file: location of the plot file to process.
            mgraph: reference to the mgraph instance we are using to build the initial KG.
        """
        
        print (f"Processing plot file: {plot_file}")
        plot = self._load_plotfile(plot_file=plot_file)

        # extract characters
        print("Extracting characters from the plot...")
        character_task = self.character_extraction_agent.run(f"""
            You will extract the characters from the attached story:
                                                                      
            Story:
            {plot}
        """)

        # extract locations
        print("Extracting locations from the plot...")
        location_task = self.location_extraction_agent.run(f"""
            You will extract the locations from the attached story:
                                                                    
            Story:
            {plot}
        """)

        character_result, location_result = await asyncio.gather(
            character_task, location_task
        )

        character_entities = character_result.output
        location_entities = location_result.output

        # extract the map triplets
        print("Extracting location connections from the plot...")
        location_triplets = self.map_creation_agent.run_sync(f"""
            You will extract the connections between locations from the attached story and locations list:
            Story:
            {plot}
            Locations:
            {json.dumps([l.model_dump() for l in location_entities], indent=2)}                                         
        """).output

        # extract the character relationship triplets
        print("Extracting character relationships from the plot...")
        character_relationship_triplets = self.character_relationship_agent.run_sync(f"""
            You will extract the relationships between characters and objects from the attached story, characters list, objects list, and locations list:
            Story:
            {plot}
            Characters:
            {json.dumps([c.model_dump() for c in character_entities], indent=2)}                                           
            Locations:
            {json.dumps([l.model_dump() for l in location_entities], indent=2)}                                           
        """).output

        # construct the initial KG
        print("Constructing the initial knowledge graph...")
        self._construct_initial_kg(location_triplets, character_relationship_triplets)

        # export the graph to a JSON file for inspection
        print("Exporting the knowledge graph to a JSON file...")
        self.graphManager.export_graph()



class AvatarAgent():

    def __init__(self, character_name: str, graphManager: MGraphManager, memory_length: int = 5):
        self.character_name = character_name
        self.graphManager = graphManager
        self.memory = deque(maxlen=memory_length)
        system_prompt = f"""
            You are an expert in determing changes to the story caused a players declared actions.
            The character the player controls is named: {character_name}
            You will be given a users input.
            You will be given tools to query the current Graph Database that represents the story world.
            You will be given tools to add/remove relationships from that Graph Database.

            Your task is to:
            * Find any changes to the world that would be caused by the players actions.
            * Update the knowledge graph as needed.
            * Generate any dialogue this character needs to move the story forward.
            * Summarize the actions you took in a user friendly manner.
        """
        
        self.agent = Agent(
            model,
            output_type=str,
            system_prompt=system_prompt
        )

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.query_by_entity_name)

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.query_by_entity_type)

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.insert_predicate)

        self.agent.tool_plain(
            docstring_format="google"
        )(graphManager.remove_predicate)

    def run(self, user_input:str) -> str:
        """
        Executes the users request.
        """
        prompt = f"""
        Please update this characters actions based on the following input:
        {user_input}

        Here is the recent conversational history:
        {self.memory}
        """
        result = self.agent.run_sync(prompt).output

        self.memory.append({
            "user_input": user_input,
            "agent_output": result
        })

        return result

class StoryTellerAgent():

    def __init__(self, graphManager: MGraphManager, story_file="my_story.txt") -> None:
        self.memory = deque(maxlen=100)
        self.story_file = story_file
        self.graphManager = graphManager
        
        self.story_teller_agent = Agent(
            model,
            output_type=str,
            system_prompt="""
            You are an expert at writing stories from the perspective of a given character.
            You will be given the name of the character.
            You will be given a summary of the each character in the stories next moves.
            You will be given the previous responses from the story teller.
            You will be given tools to query the underlying graph database representing the story for more information.
            Your task is to:
                * Synthesize the actions of each character in the story in a coherent story output, told from the
                prespective of the given story character. 
                * Output should be in "story" format.
            """
        )

        self.story_teller_agent.tool_plain(
            docstring_format="google"
        )(graphManager.query_by_entity_name)

        self.story_teller_agent.tool_plain(
            docstring_format="google"
        )(graphManager.query_by_entity_type)

    def _save_to_file(self, content: str) -> None:
        """Helper method to append a story segment to the text file."""
        try:
            with open(self.story_file, "a", encoding="utf-8") as f:
                # Appends the response followed by clean spacing for the next paragraph
                f.write(content + "\n\n")
        except IOError as e:
            print(f"Error writing to story file {self.story_file}: {e}")

    def generate_story_introduction(self, character_name: str, summary: str) -> str:
        """
        Generates the first paragraph of the shared story.
        """
        prompt = f"""
            Please generate the introduction to the story from the perspective of the following character and their actions:
            Main Character: {character_name}
            Main Character Last Action: {summary}
        """

        response = self.story_teller_agent.run_sync(prompt).output
        self.memory.append(response)
        self._save_to_file(response)

        return response

    def generate_next_story_item(self, character_name: str, character_update: str, character_updates: dict) -> str:
        """
        Generates the next segment of the story
        """
        prompt = f"""
        Please generate the next state of the story using the following parameters:

        Main Character: {character_name}
        Main Character Last Action: {character_update}

        Other Characters:
        {character_updates}

        The Story So far:
        {self.memory}
        """
        
        response = self.story_teller_agent.run_sync(prompt).output
        self.memory.append(response)

        self._save_to_file(response)
        return response

    def get_story_so_far(self):
        return self.memory
    
    def append_to_story(self, segment):
        self.memory.append(segment)

class InformationalAgent():
    def __init__(self, graphManager: MGraphManager):
        system_prompt = f"""
            You are an expert at answering questions about the state of a story.
            You will be given a graph database representing the current story.
            You will be given the most recent history of the story so far.
            You will be given tools to query the graph database.
            You will be given a user query.
            
            Your task is to:
            * Use the graph database to answer the users query.
            * Provide evidence from the graph database to support your answer.
        """

        self.informational_agent = Agent(
            model,
            output_type=str,
            system_prompt=system_prompt
        )

        self.informational_agent.tool_plain(
            docstring_format="google"
        )(graphManager.query_by_entity_name)

        self.informational_agent.tool_plain(
            docstring_format="google"
        )(graphManager.query_by_entity_type)

    
    def run(self, user_query: str) -> str:
        """
        runs a query on the informational agent
        """
        return self.informational_agent.run_sync(user_query).output


class RoutingAgent():

    def __init__(self):
        self.agent = Agent(
            model,
            output_type=str,
            system_prompt="""
                You are an expert in differentiating between questions about a story and a users attempt to add to or move the story forward.
                Your task is to:
                * Analyze if the user is asking for information about the story or game world.
                * Return "QUERY" if the user making a query, "STORY" otherwise.
            """
        )

    def run(self, user_input:str) -> bool:
        """
        Executes the RoutingAgent query

        Args:
            user_input: the users query
        """
        classification = self.agent.run_sync(user_input).output
        return "QUERY" in classification

class WorldSimulationSystem:
    
    def __init__(self, graphManager: MGraphManager):
        self.graphManager = graphManager if graphManager else MGraphManager()
        self.character_name = None

        self.avatar_agent = None
        self.character_agents = []

        self.state_summarization_agent = Agent(
            model,
            output_type=str,
            system_prompt=f"""
            You are an expert at summarizing the current state of a story from the perspective of a specific character.
            You will be given a character name.
            You will be given tools to query the knowledge graph for information about that character and their relationships to other entities in the story.
            Your task is to:
            * Summarize the current state of the story from the perspective of the given character, including the character's current situation, relationships, and any relevant events or locations, based on the information available in the knowledge graph.
            * Your summary should be concise and focus on the most important aspects of the character's perspective
            """
        )

        self.state_summarization_agent.tool_plain(
            docstring_format="google"
        )(self._fetch_entity_relationships)

        self.story_teller_agent = StoryTellerAgent(self.graphManager)
        self.information_agent = InformationalAgent(self.graphManager)
        self.routing_agent = RoutingAgent()


    def _get_all_characters(self):
        """Utility function to query the knowledge graph for all characters and their relationships."""
        return self.graphManager.query_by_entity_type("character")
    
    def get_main_character(self):
        """Utility function to query the knowledge graph for the main character of the story."""
        characters = self._get_all_characters()

        print("Available characters to experience the story from:")
        for i, character in enumerate(characters):
            print(f"{i + 1}. {character}")

        while self.character_name is None:
            character_choice = input("Please select a character by entering the corresponding number: ")
            try:
                character_index = int(character_choice) - 1
                if 0 <= character_index < len(characters):
                    self.character_name = characters[character_index]
                else:
                    print("Invalid choice. Please enter a number corresponding to one of the characters listed.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # initialize the avatar
        self.avatar_agent = AvatarAgent(self.character_name, self.graphManager)

        # initialize the character agents
        for c in characters:
            if c == self.character_name:
                continue
            character_avatar = AvatarAgent(c, self.graphManager)
            self.character_agents.append(character_avatar)

        self.guardian_agent = Agent(
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

    def _fetch_entity_relationships(self, entity_name):
        """
        Utility function to query the knowledge graph for a specific entity and its relationships.
        Args:
            entity_name (str): The name of the entity to query for.
        Returns:
            The query results from the knowledge graph.
        """
        return self.graphManager.query_by_entity_name(entity_name)

    def get_current_state_summary(self) -> str:
        """Utility function to get a summary of the current state of the story from the perspective of the selected character."""
        if self.character_name is None:
            print("No character selected. Please select a character first.")
            return ""
        
        summary = self.state_summarization_agent.run_sync(f"""
            You will summarize the current state of the story from the perspective of the character "{self.character_name}".
            Use the following tool to query the knowledge graph for information about this character and their relationships to other entities in the story:
        """).output

        self.story_teller_agent.append_to_story(summary)

        return summary

    def _run_story_agent(self, user_input: str):
        """
        Runs the story agent on the user input

        Args:
            user_input: the users prompt.
        """
        
        results = self.avatar_agent.run(f"""
                Characters next action: {user_input}
                Story So Far: 
                {self.story_teller_agent.get_story_so_far()}
            """)
          
        character_updates = {}
        for avatar in self.character_agents:
            cu = avatar.run(f"""
                    The main characters most recent actions can be summarized as:
                                
                    {results}

                    Please update on how your character would respond to this. If you're character would not be aware of these actions,
                    then please do nothing and respond with NO CHANGES

                    Here is the story so far:

                    {self.story_teller_agent.get_story_so_far()}

            """)

            print(f"Updating {avatar.character_name}...")
            character_updates[avatar.character_name] = cu
            
        next_update = self.story_teller_agent.generate_next_story_item(self.character_name, results, character_updates)
        print(f"\n{self.character_name} did the following: \n{results}")
        print(next_update)

    def _run_informational_agent(self, user_input:str):
        """
        Runs the informational agent.

        Args:
            user_input: the users prompt
        """
        response = self.information_agent.run(user_input)
        print(f"Information: \n {response}")

    def interaction_loop(self, story_file="my_story.txt"):
        """Main interaction loop for the world simulation system."""
        if self.character_name is None or self.avatar_agent is None:
            print("No character selected. Please select a character first.")
            return
        
        self.story_teller_agent.story_file = story_file
        first_paragraph = self.get_current_state_summary()
        introduction = self.story_teller_agent.generate_story_introduction(self.character_name, first_paragraph)

        print("********INTRODUCTION************")
        print(introduction)

        while True:
            user_input = input(f"\nWhat does {self.character_name} do next? (Type 'help' for options) ")

            if user_input.lower() == "help":
                print("\nOptions:")
                print("1. summary - Get a summary of the current state of the story from your character's perspective.")
                print("2. exit - Exit the simulation.")
                continue
            elif user_input.lower() == "summary":
                summary = self.get_current_state_summary()
                print(f"\nCurrent state summary from {self.character_name}'s perspective:\n{summary}")
                continue
            elif user_input.lower() == "exit" or user_input.lower() == "quit":
                print("Exiting the simulation. Goodbye!")
                break

            guardrails_check = self.guardian_agent.run_sync(f"Review the following input and determine if it contains any content that would be considered inappropriate based on the guidelines you follow. If it does, respond with 'CANNOT CONSTRUCT GRAPH'. If it does not, respond with 'GRAPH CONSTRUCTION APPROVED'. Plot: {user_input}").output 
            if "CANNOT UPDATE" in guardrails_check:
                print("I am afraid this request violates my ethical and safety guidelines.")
                continue
            
            is_query = self.routing_agent.run(user_input)

            if is_query:
                self._run_informational_agent(user_input)
            else:
                self._run_story_agent(user_input)


########################################################################
# UTLITY FUNCTIONS
########################################################################

# Initialize MGraphManager
graph_manager = MGraphManager()

def get_input_args():
    """
    Retrieves and parses the command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_file', type=str, default='plot.md', help='Path to the plot file')
    parser.add_argument('--construct_graph', action='store_true', help='Whether to construct the knowledge graph')
    parser.add_argument('--graph_file', type=str, default='knowledge_graph.json', help='Path to the knowledge graph file (if loading an existing graph)')
    parser.add_argument('--story_file', type=str, default='my_story.txt', help='Path to the file your story will be saved to')
    
    return parser.parse_args()


########################################################################
# MAIN LOOP
########################################################################

def main():
    # Get command-line arguments
    args = get_input_args()

    # Initialize WorldGenerationSystem with the graph manager
    world_generation_system = WorldGenerationSystem(graph_manager)
    world_simulation_system = WorldSimulationSystem(graph_manager)

    # Process the plot and construct the KG if the flag is set
    if args.construct_graph:
        asyncio.run(world_generation_system.process_plot(plot_file=args.plot_file))
    else:
        print("Skipping graph construction. Use --construct_graph flag to enable it.")


    graph_manager.restore_graph(graph_file=args.graph_file)
    world_simulation_system.get_main_character()
    world_simulation_system.interaction_loop(story_file=args.story_file)


if __name__ == "__main__":
    main()