#!/usr/bin/env python
# coding: utf-8

# This is a starter notebook for the project, you'll have to import the libraries you'll need, you can find a list of the ones available in this workspace in the requirements.txt file in this workspace. 

# In[40]:


import os

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"


# In[44]:


# import libraries

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, CombinedMemory, ChatMessageHistory
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain


# In[ ]:





# # Setting Up the Python Application

# In[45]:


get_ipython().system('ls')


# In[46]:


get_ipython().system('pip install -r requirements.txt')


# In[47]:


# Import OpenAI library
import openai
# from openai.embeddings_utils import get_embedding
# Set the base URL for the OpenAI API (Vocareum's endpoint)
openai.api_base = "https://openai.vocareum.com/v1"
# Set the API key (Needs to be provided for authentication)
openai.api_key =  "voc-1195158280126677390712267cecde3101f94.66438984"


# In[66]:


# Batch size and Model Name for processing
BATCH_SIZE = 64
model_name = 'gpt-3.5-turbo'


# In[49]:


prompt = '''
Generate 10 real estate listings. Each listing should include:
- Neighborhood
- Price
- Bedrooms
- Bathrooms
- House Size
- Description
- Neighborhood Description

Follow the structure of this example:

Neighborhood: Green Oaks
Price: $800,000
Bedrooms: 3
Bathrooms: 2
House Size: 2,000 sqft

Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.

Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.

Could you please generate it as comma sperated file as rows
'''


# In[50]:


response = openai.Completion.create(
        model=COMPLETION_MODEL_NAME,
        prompt=prompt,
        max_tokens=244  
    )


# In[78]:


response = """
Neighborhood,Price,Bedrooms,Bathrooms,House Size,Description,Neighborhood Description
Green Oaks,"$800,000 ",3,2,"2,000 sqft","Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family.","Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze."
Willow Creek,"$925,000 ",4,3,"2,700 sqft","This spacious 4-bedroom, 3-bathroom home in Willow Creek is perfect for families seeking modern amenities and serene surroundings. With vaulted ceilings, a large open-concept kitchen, and a beautifully landscaped backyard, this home invites comfort and relaxation. The master suite includes a private balcony and a spa-inspired bath.","Willow Creek is known for its scenic walking trails, family-friendly parks, and strong school system. The community is active and tight-knit, with regular events at the local community center and easy access to shopping and dining."
Bayview Heights,"$1,100,000 ",5,4,"3,400 sqft","Nestled on a hillside with panoramic ocean views, this 5-bedroom home in Bayview Heights offers luxury coastal living. Features include a chef's kitchen, hardwood flooring, and a spacious deck perfect for entertaining. Floor-to-ceiling windows bring in abundant natural light.","Bayview Heights is a prestigious neighborhood offering breathtaking views, upscale amenities, and close proximity to beaches and marinas. The community is known for its peaceful vibe and scenic beauty."
Elmwood Grove,"$675,000 ",3,2,"2,100 sqft","This beautifully updated ranch-style home in Elmwood Grove combines charm and functionality. The open floor plan connects the living room, dining room, and kitchen, making it ideal for entertaining. A spacious backyard with a pergola completes the picture.","Elmwood Grove is a quiet, leafy suburb with top-rated schools, cozy cafés, and local boutiques. Residents love the sense of community and access to green spaces."
Stonebridge,"$800,000 ",4,3,"2,600 sqft","Located in the heart of Stonebridge, this elegant 4-bedroom home offers modern comforts and classic appeal. Enjoy a gourmet kitchen with granite countertops, a formal dining room, and a finished basement ideal for a media room.","Stonebridge offers the perfect balance of city access and suburban peace. Its tree-lined streets, excellent public transportation, and community events make it a top choice for families and professionals."
Cedar Hills,"$725,000 ",3,2.5,"2,300 sqft","This charming two-story home in Cedar Hills features a welcoming front porch, a large family room with a fireplace, and a bright kitchen with a breakfast nook. The backyard includes a firepit and patio, perfect for entertaining.","Cedar Hills is a friendly and walkable neighborhood with top schools, neighborhood parks, and weekend farmers markets. It’s loved for its strong sense of community and natural surroundings."
Lakeview Estates,"$990,000 ",4,3.5,"3,100 sqft","Enjoy lakeside luxury in this beautifully designed 4-bedroom home in Lakeview Estates. With large windows showcasing water views, a modern kitchen, and a private dock, this home is an entertainer’s dream.","Lakeview Estates is a peaceful and exclusive lakefront community. Residents enjoy boating, fishing, and walking along scenic trails. It’s ideal for those who love waterfront living and privacy."
Maple Heights,"$585,000 ",3,2,"1,900 sqft","Cozy yet stylish, this 3-bedroom bungalow in Maple Heights features a wood-burning fireplace, original hardwood floors, and a sun-drenched breakfast nook. The backyard garden adds charm and tranquility.",Maple Heights is a historic neighborhood filled with character homes and mature trees. Its close-knit community hosts seasonal events and supports local businesses and artisans.
Riverbend,"$870,000 ",4,3,"2,800 sqft","Welcome to this modern 4-bedroom home in Riverbend. Features include an open floor plan, floor-to-ceiling windows, a gourmet kitchen, and a large deck that backs onto greenbelt space for added privacy.","Riverbend is a nature-forward neighborhood with access to river trails, kayaking spots, and birdwatching areas. It appeals to active families and outdoor enthusiasts."
Parkside Village,"$760,000 ",3,2.5,"2,400 sqft","This stylish Parkside Village home offers open-concept living, an upstairs loft, and a large kitchen with a center island. The master suite features a walk-in closet and spa bathroom.","Parkside Village is a modern community known for its beautiful parks, inclusive vibe, and weekend street markets. It's a popular spot for young professionals and growing families."
Highland Meadows,"$1,200,000 ",5,4,"3,800 sqft","This luxurious estate in Highland Meadows includes five bedrooms, a formal living and dining area, a home office, and a beautifully landscaped yard with a pool and hot tub. Perfect for elegant entertaining.","Highland Meadows is an affluent neighborhood with manicured streets, private schools, and a country club. It's known for privacy, security, and high-end amenities."
"""


# In[79]:


mess = response['choice'][0]['message']


# In[61]:


# The write csv text

with open('dataset.csv', 'w') as f:
    f.write(mess)


# In[65]:


# read and load the csv file that store homes data
loader = CSVLoader(file_path='dataset.csv')
docs = loader.load()
docs


# In[68]:


# Load documents
llm = OpenAI(model_name=model_name, temperature=0, max_tokens=2000)


# In[69]:


# Use a Text Splitter to split the documents into chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = splitter.split_documents(docs)
split_docs


# In[70]:


# Populate the vector database with the chunks
db = Chroma.from_documents(split_docs, embeddings)


# # Building the User Preference Interface

# In[71]:


# The buyer preferences in questions and answers example:

questions = [   
                "How big do you want your house to be?" 
                "What are 3 most important things for you in choosing this property?", 
                "Which amenities would you like?", 
                "Which transportation options are important to you?",
                "How urban do you want your neighborhood to be?",   
            ]

answers = [
    "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and convenient shopping options.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
]


# In[72]:


# create a chat with the user and 
history = ChatMessageHistory()
history.add_user_message(f"""You are an AI sales assistant. Based on the answers to the following {len(questions)} questions, you will recommend a home that best fits the user's preferences. Ask the user each question and consider their responses to suggest the most suitable property.""")
for i in range(len(questions)):
    history.add_ai_message(questions[i])
    history.add_user_message(questions[i])


# In[73]:


# summarize a chat
max_rating = 5
summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="recommendation_summary",
    input_key="input",
    buffer=f"The user answered {len(questions)} personalized questions. Use their answers to rate, from 1 to {max_rating}, how well a home recommendation matches their preferences.",
    return_messages=True
)


# In[74]:


from typing import Any, Dict

class MementoBufferMemory(ConversationBufferMemory):
    # Save the context by extracting input-output pairs and adding AI's response to chat memory
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        # Extract input and output strings from the given inputs and outputs
        input_str, output_str = self._get_input_output(inputs, outputs)
        
        # Add the AI-generated output to the chat memory
        self.chat_memory.add_ai_message(output_str)

# Create an instance of MementoBufferMemory to store and manage conversational history
conversational_memory = MementoBufferMemory(
    chat_memory=history,  # Use the existing history object to manage conversation
    memory_key="questions_and_answers",  # Key for storing question-answer pairs
    input_key="input"  # Key for accessing user input data
)

# Combine conversational memory and summary memory into a single memory object
memory = CombinedMemory(memories=[conversational_memory, summary_memory])


# # Personalizing Listing Descriptions

# In[75]:


# Initialize an empty list to store the user's responses from the conversation buffer
user_responses = []

# Loop through the messages in the conversational memory buffer
for message in conversational_memory.buffer_as_messages:
    # Check if the message type is 'human' (i.e., user response)
    if message.type == "human":
        # Append the content of the user's response to the list
        user_responses.append(message.content)

# Combine all the user's responses into a single string (user preferences)
user_preferences = " ".join(user_responses)

# Use the combined user preferences to search for similar documents in the database
similar_docs = db.similarity_search(user_preferences, k=5)

# Format the top 5 similar documents as recommended listings
recommended_listings = "\n\n---------------------\n\n".join([doc.page_content for doc in similar_docs])


# In[76]:


# Define a template for the friendly conversation between the human and the AI Real Estate Agent
RECOMMENDER_TEMPLATE = """
The following is a friendly conversation between a human and an AI Real Estate Agent. 
The AI follows human instructions and provides home ratings for a human based on the home preferences. 

Summary of Recommendations:
{recommendation_summary}

Buyer's Preferences Q&A:
{questions_and_answers}

Recommended Listings:
{recommended_listings}

Human: {input}
AI:"""

# Create a partially populated PromptTemplate using the RECOMMENDER_TEMPLATE, 
# with the 'recommended_listings' variable inserted into the template
PROMPT = PromptTemplate.from_template(RECOMMENDER_TEMPLATE).partial(recommended_listings=recommended_listings)

# Initialize the ConversationChain with the language model (llm), memory, and the defined prompt
recommender = ConversationChain(
    llm=llm,           # The language model used for conversation
    verbose=True,       # Enable verbose output for debugging
    memory=memory,      # Memory object to maintain conversational context
    prompt=PROMPT       # The prompt template used for generating conversation
)


# # Deliverables and Testing

# In[77]:


# Define an augmented query to generate personalized descriptions for the top 5 listings
augmented_query = """
Now score (0-5) each of the 10 listings based on the buyer's preferences. Format the output as follows:

Home Match Score: [Score]                   # Score based on how well the listing matches buyer's preferences
Neighborhood: [Neighborhood]               # Neighborhood description tailored to the buyer's preferences
Price: [Price]                             # Price of the listing
Bedrooms: [Bedrooms]                       # Number of bedrooms in the listing
Bathrooms: [Bathrooms]                     # Number of bathrooms in the listing
Size sqft: [Size sqft]                     # Size of the listing in square feet
Description: [Personalize both the description and the neighborhood description of the listing based on buyer's preferences. Make sure the modified description is unique, appealing, and tailored to the buyer's provided preferences but keep the modified description factual]
"""

# Generate personalized recommendation by passing the augmented query into the recommender
personalized_recommendation = recommender.predict(input=augmented_query)

# Print the generated personalized recommendation
print(personalized_recommendation)


# In[ ]:



