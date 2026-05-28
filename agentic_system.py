# Interactive Fiction Agentic System
from pydantic_ai import Agent
from pydantic import BaseModel, Field

class PlotEntity(BaseModel):
    name: str = Field(description="Name of the entity extracted from the plot")
    type: str = Field(description="The type of entity")
    description: str = Field(description="Description of the extracted entity")


# This system will work in two phases:

# 1. Startup Phase

# Load the plot file into context.
def load_plot_file(filename="plot.md"):
    """
    loads the plot file from which the initial KG will be constructed.

    Args:
        filename: filename for the plot file.
    Returns:
        contents of the plot file.
    """
    with open(filename, 'r') as file:
        content = file.read()
        return content




# Load the Entity Extraction Competency Questions (CQ's)
entity_extraction_cq = [
    "Who are the characters of this story?", # Character Entities
    "What locations are in this story?", # Location Entities
    "What important objects are in this story?", # Object Entities
    "What important plot points exist in this story?" # Plot Entities.
]

# Execute the Entity Extraction CQ Prompt.
# Combine Plot + Entity CQ Prompt Results into a new Context.
# Execute the Relationship Creation Loop.

# 2. Interaction Phase

# Find starting location
# Show user description.

# Begin Agentic Loop
#   User provides response.
#   Agent loads relevants parts of the KG
#   Updates KG as needed.
#   Agent sends back response.
