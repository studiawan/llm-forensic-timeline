from dotenv import load_dotenv
import openai
import os
import json

# Load .env file
load_dotenv()

# Get the API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# OpenAI client initialization
client = openai.OpenAI()

# list of tasks
tasks = ["event-summarization"]

for task in tasks:
    # Set the directory containing files and prompts
    current_directory = os.getcwd()
    dataset_directory = os.path.join(current_directory, f"dataset/{task}/")
    prompts_directory = os.path.join(current_directory, f"prompt/{task}/")

    # Function to read prompts from a file
    for prompt_file in os.listdir(prompts_directory):
        # Construct the full path to the JSON evaluation file
        json_path = os.path.join(prompts_directory, prompt_file)
        print("JSON Path:", json_path)

        if os.path.isfile(json_path):
            # read json file
            with open(json_path, "r") as f:
                # Load the JSON content
                prompt = json.load(f)

                # Extracting information from the prompt
                task = prompt['task']
                type_type = prompt['type']
                role = prompt['role']
                instructions = prompt['instructions']
                output_format = prompt['output_format']

                # loop through goals and files
                for file_name, goal in prompt['goals'].items():
                    # Construct the full path to the CSV forensic timeline file
                    csv_path = os.path.join(dataset_directory, file_name)
                    print(f"File Name: {csv_path}, Goal: {goal}")

                    # uploading file to OpenAI
                    try:
                        print(f"Uploading forensic timeline in CSV: {csv_path}...")
                        file_object = client.files.create(
                            file=open(csv_path, "rb"),
                            purpose='assistants'
                        )
                        print(f"File uploaded successfully. File ID: {file_object.id}")
                
                    except FileNotFoundError:
                        print(f"Error: The file '{csv_path}' was not found.")
                        exit()

                    # create OpenAI assistant and thread
                    print("Creating Assistant and a new conversation Thread...")
                    assistant = client.beta.assistants.create(
                        name="Forensic timeline analyst",
                        instructions=role,
                        model="gpt-4o",
                        tools=[{"type": "code_interpreter"}],
                        tool_resources={"code_interpreter": {"file_ids": [file_object.id]}}
                    )
                    thread = client.beta.threads.create()
                    print(f"Assistant ({assistant.id}) and Thread ({thread.id}) created.")

                    # create list of prompts
                    prompts = [
                        role,
                        f"{goal}\n\n{instructions}\n\n{output_format}"
                    ]

                    for prompt in prompts:
                        print(f"User: {prompt}")

                        # Add the user's message to the existing thread
                        client.beta.threads.messages.create(
                            thread_id=thread.id,
                            role="user",
                            content=prompt
                        )

                        # Create and run the assistant on the thread
                        run = client.beta.threads.runs.create(
                            thread_id=thread.id,
                            assistant_id=assistant.id
                        )

                        # Wait for the run to complete
                        print("Assistant is thinking...")
                        while run.status in ['queued', 'in_progress']:
                            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

                        # Process response and download file if needed
                        if run.status == 'completed':
                            print("Assistant completed successfully.")
                            # Here you can handle the response as needed
                            messages = client.beta.threads.messages.list(thread_id=thread.id, limit=1)
                            assistant_message = messages.data[0]
                            
                            # Print the text response from the assistant
                            text_content = assistant_message.content[0].text
                            print(f"Assistant: {text_content.value}")
                            
                            # Check for file annotations in the response
                            annotations = text_content.annotations
                            if annotations:
                                for annotation in annotations:
                                    if annotation.type == 'file_path':
                                        file_id = annotation.file_path.file_id
                                        file_name_in_annotation = annotation.text.split('/')[-1]
                                        print(f"\nAssistant generated a file: '{file_name_in_annotation}' ({file_id})")
                                        
                                        # Download the file content
                                        print("Downloading file...")
                                        file_content_response = client.files.content(file_id)
                                        file_content = file_content_response.read()

                                        # Define the directory to save results
                                        results_directory = os.path.join(os.getcwd(), "results")
                                        # Create the directory if it doesn't exist
                                        os.makedirs(results_directory, exist_ok=True)  
                                        
                                        # Save the file in the results directory
                                        file_path = os.path.join(results_directory, file_name_in_annotation)
                                        with open(file_path, "wb") as f:
                                            f.write(file_content)
                                        print(f"File '{file_name_in_annotation}' downloaded successfully.")
                        else:
                            print(f"Run failed with status: {run.status}")
                            break # Exit the loop if a run fails
                            
                        
                    