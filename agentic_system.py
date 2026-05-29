########################################################################
# Interactive Fiction Agentic System
########################################################################

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


# KG Construction Agents
character_extraction_agent = Agent(
    model,
    output_type=list[PlotEntity],
    system_prompt=f"""
    You are an expert in identifying characters in a story.
    You will be given a story.
    Your task is to:
    * identify all primary characters in the story.
    * generate a description of the character from story.
    """
)

location_extraction_agent = Agent(
    model,
    output_type=list[PlotEntity],
    system_prompt=f"""
    You are an expert in identifying locations in a story.
    You will be given a story.
    Your task is to:
    * identify all primary locations in the story.
    * generate a description of the location based on the story.
    """
)

map_creation_agent = Agent(
    model,
    output_type=list[GraphTriplet],
    system_prompt=f"""
    You are an expert at mapping out the connections between locations from a story.
    You will be given a story.
    You will be given a list of locations from that story.
    Your task is to:
    * Construct a collection of triplets where each triplet represents a connection between two locations in the story.
    * Assign the "connects_to" label as the predicate for each triplet.
    """
)

# Character / Object Relation Agent
character_relationship_agent = Agent(
    model,
    output_type=list[GraphTriplet],
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


# Fiction Simulation Agents
# Avatar Agent
# Character Agent
# World Agent



