# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "requests",
#     "db-sqlite3",
#     "scipy",
#     "python-dateutil",
#     "pybase64",
#     "duckdb",
#     "pandas",
#     "numpy",
#     "pillow",
#     "gitpython",
#     "markdown",
#     "SpeechRecognition",
# ]
# ///


from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn

app = FastAPI()


import subprocess
import os
from pathlib import Path
import json
import requests
import sqlite3
from scipy.spatial.distance import cosine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


OPENAI_API_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")


def get_nested_value(data, keys):
    """Recursively retrieves a nested value from a dictionary using a list of keys/indexes."""
    try:
        for key in keys:
            if isinstance(data, list) and isinstance(key, int):  # Handle list indexing
                data = data[key]
            elif isinstance(data, dict) and key in data:  # Handle dictionary keys
                data = data[key]
            else:
                print(data)
                raise KeyError(f"Key '{key}' not found.")
        return data
    except (KeyError, IndexError, TypeError) as e:
        logging.error(f"Error accessing nested key {keys}: {e}")
        raise HTTPException(status_code=500, detail=f"Error accessing nested key {keys}: {e}")


def handle_openai_api_request(url, data, headers=None, method='post', json_key=None, **kwargs):
    try:
        if not headers:
            headers = {
                "Authorization": f"Bearer {AIPROXY_TOKEN}",
                "Content-Type": "application/json"
            }
        if method.lower() == 'post':
            response = requests.post(url, headers=headers, json=data, **kwargs)
        elif method.lower() == 'get':
            response = requests.get(url, headers=headers, params=data, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")

        try:
            response.raise_for_status()  # Raises an error for HTTP 4xx/5xx responses
        except requests.exceptions.HTTPError as e:
            logging.error(f"API response: {response.text}")
            raise e

        response_json = response.json()

        if json_key:
            if isinstance(json_key, str):
                json_key = json_key.split('.')  # Convert dot notation to list
            elif not isinstance(json_key, list):
                raise ValueError("json_key must be a string (dot notation) or a list of keys.")
            return get_nested_value(response_json, [int(k) if k.isdigit() else k for k in json_key])  # Convert list indexes to integers
        return response_json

    except requests.exceptions.RequestException as e:
        logging.error(f"Request to OpenAI API failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request to OpenAI API failed: {str(e)}")
    except json.JSONDecodeError:
        logging.error("Failed to parse OpenAI API response as JSON")
        raise HTTPException(status_code=500, detail="Failed to parse OpenAI API response as JSON")


def install_uv_and_run_datagen(user_email: str):
    """
    Installs the 'uv' package if required and runs the datagen.py script with the user's email as the only argument.
    
    Args:
        user_email (str): The user's email to be passed as an argument to the datagen.py script.
    """
    try:
        process = subprocess.Popen(
            ["uv", "run", "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py", user_email],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Error: {stderr}")
        return stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error: {e.stderr}")


def format_with_prettier(file_path: str="/data/format.md"):
    """
    Formats the given file using prettier@3.4.2 and updates it in place.
    
    Args:
        file_path (str): The path to the file to be formatted.
    """
    try:
        subprocess.run(["npx", "prettier@3.4.2", "--write", file_path], check=True)
        return (f"Formatted {file_path} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error formatting {file_path}: {e}")


def count_weekdays(input_file: str="/data/dates.txt", output_file: str="/data/dates-weekdays.txt", weekday: str="Wednesday"):
    """
    Counts the number of specified weekdays in the given file and writes the count to another file.
    
    Args:
        input_file (str): The path to the file containing the list of dates.
        output_file (str): The path to the file where the count of specified weekdays will be written.
        weekday (str): The name of the weekday to count (default is "Wednesday").
    """
    try:
        with open(input_file, 'r') as file:
            dates = file.readlines()
        
        weekday_count = 0
        target_weekday = parser.parse(weekday).weekday()
        
        for date in dates:
            try:
                parsed_date = parser.parse(date.strip())
                if parsed_date.weekday() == target_weekday:
                    weekday_count += 1
            except ValueError:
                print(f"Skipping invalid date format: {date.strip()}")
        
        with open(output_file, 'w') as file:
            file.write(str(weekday_count))
        
        return (f"Counted {weekday_count} {weekday}s and wrote to {output_file} successfully.")
    except Exception as e:
        print(f"Error processing files: {e}")


def write_recent_logs(log_dir: str="/data/logs", output_file: str="/data/logs-recent.txt", count: int=10):
    """
    Writes the first line of the most recent .log files to an output file.
    
    Args:
        log_dir (str): The directory containing the log files.
        output_file (str): The path to the output file.
        count (int): The number of recent log files to process.
    """
    try:
        log_files = sorted(Path(log_dir).glob("*.log"), key=os.path.getmtime, reverse=True)[:count]
        
        with open(output_file, 'w') as out_file:
            for log_file in log_files:
                with open(log_file, 'r') as file:
                    first_line = file.readline().strip()
                    out_file.write(first_line + "\n")
        
        return (f"Wrote the first line of the {count} most recent log files to {output_file} successfully.")
    except Exception as e:
        print(f"Error processing log files: {e}")


def create_markdown_index(docs_dir: str="/data/docs", index_file: str="/data/docs/index.json"):
    """
    Creates an index of Markdown files mapping filenames to their first H1 title.
    
    Args:
        docs_dir (str): The directory containing the Markdown files.
        index_file (str): The path to the index file to be created.
    """
    try:
        index = {}
        for md_file in Path(docs_dir).rglob("*.md"):
            with open(md_file, 'r') as file:
                for line in file:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        index[str(md_file.relative_to(docs_dir))] = title
                        break
        
        with open(index_file, 'w') as out_file:
            json.dump(index, out_file, indent=4)
        
        return (f"Created index file at {index_file} successfully.")
    except Exception as e:
        print(f"Error creating index file: {e}")


def extract_sender_email(input_file: str="/data/email.txt", output_file: str="/data/email-sender.txt"):
    """
    Extracts the sender's email address from an email message using an LLM and writes it to a file.
    
    Args:
        input_file (str): The path to the file containing the email message.
        output_file (str): The path to the file where the sender's email address will be written.
    """
    try:
        with open(input_file, 'r') as file:
            email_content = file.read()
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a function classifier that extracts structured parameters from queries."},
                {"role": "user", "content": f"Extract the sender's email address from the following email message:\n\n{email_content}\n\nSender's email address:"}
            ],
            "max_tokens": 10
        }
        
        sender_email = handle_openai_api_request(
            url=f"{OPENAI_API_BASE_URL}/chat/completions",
            data=data,
            json_key="choices.0.message.content"
        ).strip()
        
        with open(output_file, 'w') as file:
            file.write(sender_email)
        
        return (f"Extracted sender's email address and wrote to {output_file} successfully.")
    except Exception as e:
        print(f"Error extracting sender's email address: {e}")


import re


def extract_credit_card_number(image_path: str="/data/credit-card.png", output_file: str="/data/credit-card.txt"):
    """
    Extracts the credit card number from an image using an LLM and writes it to a file without spaces.
    
    Args:
        image_path (str): The path to the image containing the credit card number.
        output_file (str): The path to the file where the credit card number will be written.
    """
    try:
        with open(image_path, 'rb') as file:
            image_data = file.read()
        
        encoded_image_data = base64.b64encode(image_data).decode('utf-8')
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all the numbers in the image. Return only the longest 16 digit number."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "detail": "high",
                                "url": f"data:image/png;base64,{encoded_image_data}"
                            }
                        }
                    ]
                }
            ]
        }
        
        response = handle_openai_api_request(
            url=f"{OPENAI_API_BASE_URL}/chat/completions",
            data=data,
            json_key='choices.0.message.content'
        ).strip().replace(' ', '')

        match = re.search(r"\d{8,}", response)

        if match:
            credit_card_number = match.group()
        else:
            print(f"No 16-digit number found {response}")
        
        with open(output_file, 'w') as file:
            file.write(credit_card_number)
        
        return (f"Extracted credit card number {credit_card_number} and wrote to {output_file} successfully.")
    except Exception as e:
        print(f"Error extracting credit card number: {e}")


def find_most_similar_comments(input_file: str="/data/comments.txt", output_file: str="/data/comments-similar.txt"):
    """
    Finds the most similar pair of comments using embeddings and writes them to a file.
    
    Args:
        input_file (str): The path to the file containing the list of comments.
        output_file (str): The path to the file where the most similar pair of comments will be written.
    """
    try:
        with open(input_file, 'r') as file:
            comments = [line.strip() for line in file.readlines()]
        
        data = {
            "model": "text-embedding-3-small",
            "input": comments
        }
        
        response_json = handle_openai_api_request(
            url=f"{OPENAI_API_BASE_URL}/embeddings",
            data=data
        )
        
        embeddings = [item["embedding"] for item in response_json["data"]]
        
        similarity_matrix = [[cosine(embeddings[i], embeddings[j]) for j in range(len(embeddings))] for i in range(len(embeddings))]
        min_dist = float('inf')
        most_similar_pair = (None, None)
        
        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                if similarity_matrix[i][j] < min_dist:
                    min_dist = similarity_matrix[i][j]
                    most_similar_pair = (comments[i], comments[j])
        
        with open(output_file, 'w') as file:
            file.write(most_similar_pair[0] + "\n")
            file.write(most_similar_pair[1] + "\n")
        
        return (f"Found most similar comments and wrote to {output_file} successfully.")
    except Exception as e:
        print(f"Error finding most similar comments: {e}")


def calculate_gold_ticket_sales(db_path: str="/data/ticket-sales.db", output_file: str="/data/ticket-sales-gold.txt"):
    """
    Calculates the total sales of all the items in the "Gold" ticket type and writes the number to a file.
    
    Args:
        db_path (str): The path to the SQLite database file.
        output_file (str): The path to the file where the total sales will be written.
    """

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        total_sales = cursor.fetchone()[0]
        
        with open(output_file, 'w') as file:
            file.write(str(total_sales))
        
        return (f"Calculated total sales for 'Gold' tickets and wrote to {output_file} successfully.")
    except Exception as e:
        print(f"Error calculating total sales: {e}")
    finally:
        if conn:
            conn.close()


def sort_contacts(input_file: str="/data/contacts.json", output_file: str="/data/contacts-sorted.json"):
    """
    Sorts the array of contacts by last_name, then first_name, and writes the result to a file.
    
    Args:
        input_file (str): The path to the file containing the list of contacts.
        output_file (str): The path to the file where the sorted list of contacts will be written.
    """
    try:
        with open(input_file, 'r') as file:
            contacts = json.load(file)
        
        sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))
        
        with open(output_file, 'w') as file:
            json.dump(sorted_contacts, file, indent=4)
        
        return (f"Sorted contacts and wrote to {output_file} successfully.")
    except Exception as e:
        print(f"Error sorting contacts: {e}")


tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_data_from_api",
            "description": "Fetch data from the given API URL and save it to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_url": {
                        "type": "string",
                        "description": "The URL of the API to fetch data from."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the fetched data will be saved."
                    }
                },
                "required": ["api_url", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "description": "Sort the array of contacts by last_name, then first_name, and write the result to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "The path to the file containing the list of contacts."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the sorted list of contacts will be written."
                    }
                },
                "required": ["input_file", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_with_prettier",
            "description": "Format the given file using prettier@3.4.2 and update it in place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to be formatted."
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_weekdays",
            "description": "Count the number of specified weekdays in the given file and write the count to another file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "The path to the file containing the list of dates."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the count of specified weekdays will be written."
                    },
                    "weekday": {
                        "type": "string",
                        "description": "The name of the weekday to count (e.g., 'Wednesday')."
                    }
                },
                "required": ["input_file", "output_file", "weekday"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_recent_logs",
            "description": "Write the first line of the most recent .log files to an output file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_dir": {
                        "type": "string",
                        "description": "The directory containing the log files."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the output file."
                    },
                    "count": {
                        "type": "integer",
                        "description": "The number of recent log files to process."
                    }
                },
                "required": ["log_dir", "output_file", "count"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_markdown_index",
            "description": "Create an index of Markdown files mapping filenames to their first H1 title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "docs_dir": {
                        "type": "string",
                        "description": "The directory containing the Markdown files."
                    },
                    "index_file": {
                        "type": "string",
                        "description": "The path to the index file to be created."
                    }
                },
                "required": ["docs_dir", "index_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_sender_email",
            "description": "Extract the sender's email address from an email message and write it to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "The path to the file containing the email message."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the sender's email address will be written."
                    }
                },
                "required": ["input_file", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_credit_card_number",
            "description": "Extract the credit card number from an image and write it to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "The path to the image containing the credit card number."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the credit card number will be written."
                    }
                },
                "required": ["image_path", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_most_similar_comments",
            "description": "Find the most similar pair of comments and write them to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "The path to the file containing the list of comments."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the most similar pair of comments will be written."
                    }
                },
                "required": ["input_file", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_gold_ticket_sales",
            "description": "Calculate the total sales of all the items in the 'Gold' ticket type and write the number to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "The path to the SQLite database file."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the total sales will be written."
                    }
                },
                "required": ["db_path", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "install_uv_and_run_datagen",
            "description": "Install the 'uv' package if required and run the script with the given email as argument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_email": {
                        "type": "string",
                        "description": "The user's email to be passed as an argument to the datagen.py script."
                    }
                },
                "required": ["user_email"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


from fastapi import HTTPException
from dateutil import parser
import base64
import pandas as pd
import duckdb
from PIL import Image
import speech_recognition as sr
from git import Repo
import markdown
import uuid


def fetch_data_from_api(api_url: str, output_file: str):
    """
    Fetches data from the given API URL and saves it to a file.
    
    Args:
        api_url (str): The URL of the API to fetch data from.
        output_file (str): The path to the file where the fetched data will be saved.
    """
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raises an error for HTTP 4xx/5xx responses

        with open(output_file, 'w') as file:
            file.write(response.text)
        
        return (f"Fetched data from {api_url} and saved to {output_file} successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")


def clone_repo_and_commit(repo_url: str, commit_message: str="Update file", file_to_modify: str="README.md", new_content: str="New content"):
    """
    Clones a git repository, modifies a file, and makes a commit using GitPython.
    
    Args:
        repo_url (str): The URL of the git repository to clone.
        commit_message (str, optional): The commit message. Defaults to "Update file".
        file_to_modify (str, optional): The path to the file to modify within the cloned repository. Defaults to "README.md".
        new_content (str, optional): The new content to write to the file. Defaults to "New content".
    """
    try:
        repo_dir = "/tmp/repo"
        
        # Clone the repository
        repo = Repo.clone_from(repo_url, repo_dir)
        
        # Modify the file
        file_path = Path(repo_dir) / file_to_modify
        with open(file_path, 'w') as file:
            file.write(new_content)
        
        # Commit the changes
        repo.index.add([file_to_modify])
        repo.index.commit(commit_message)
        
        return (f"Cloned repo from {repo_url}, modified {file_to_modify}, and committed with message '{commit_message}' successfully.")
    except Exception as e:
        print(f"Error during git operations: {e}")

tools.append(
    {
        "type": "function",
        "function": {
            "name": "clone_repo_and_commit",
            "description": "Clone a git repository, modify a file, and make a commit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "The URL of the git repository to clone."
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "The commit message. Defaults to 'Update file'."
                    },
                    "file_to_modify": {
                        "type": "string",
                        "description": "The path to the file to modify within the cloned repository. Defaults to 'README.md'."
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The new content to write to the file. Defaults to 'New content'."
                    }
                },
                "required": ["repo_url", "commit_message", "file_to_modify", "new_content"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)

def run_sql_query(db_path: str, query: str, output_file: str):
    """
    Runs a SQL query on a SQLite or DuckDB database and writes the result to a file.
    
    Args:
        db_path (str): The path to the database file.
        query (str): The SQL query to run.
        output_file (str): The path to the file where the query result will be written.
    """
    try:
        if db_path.endswith(".duckdb"):
            conn = duckdb.connect(db_path)
        else:
            conn = sqlite3.connect(db_path)

        df = pd.read_sql_query(query, conn)
        df.to_csv(output_file, index=False)

        return (f"Ran query on {db_path} and wrote result to {output_file} successfully.")
    except Exception as e:
        print(f"Error running SQL query: {e}")
    finally:
        if conn:
            conn.close()

tools.append(
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": "Run a SQL query on a SQLite or DuckDB database and write the result to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "The path to the database file."
                    },
                    "query": {
                        "type": "string",
                        "description": "The SQL query to run."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the query result will be written."
                    }
                },
                "required": ["db_path", "query", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)


def scrape_website(url: str, output_file: str):
    """
    Scrapes data from the given website URL and saves it to a file.
    
    Args:
        url (str): The URL of the website to scrape data from.
        output_file (str): The path to the file where the scraped data will be saved.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an error for HTTP 4xx/5xx responses

        with open(output_file, 'w') as file:
            file.write(response.text)
        
        return (f"Scraped data from {url} and saved to {output_file} successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error scraping website: {e}")

tools.append(
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": "Scrape data from the given website URL and save it to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the website to scrape data from."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the scraped data will be saved."
                    }
                },
                "required": ["url", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)

def compress_or_resize_image(input_image_path: str, output_image_path: str, max_width: int = None, max_height: int = None, quality: int = 85):
    """
    Compresses or resizes an image and saves it to a file.
    
    Args:
        input_image_path (str): The path to the input image file.
        output_image_path (str): The path to the output image file.
        max_width (int, optional): The maximum width of the output image. Defaults to None.
        max_height (int, optional): The maximum height of the output image. Defaults to None.
        quality (int, optional): The quality of the output image (1-100). Defaults to 85.
    """

    try:
        with Image.open(input_image_path) as img:
            # Resize image if max dimensions are provided
            if max_width or max_height:
                img.thumbnail((max_width, max_height), Image.ANTIALIAS)
            
            # Save the image with the specified quality
            img.save(output_image_path, quality=quality, optimize=True)
        
        return (f"Compressed or resized image and saved to {output_image_path} successfully.")
    except Exception as e:
        print(f"Error compressing or resizing image: {e}")

tools.append(
    {
        "type": "function",
        "function": {
            "name": "compress_or_resize_image",
            "description": "Compress or resize an image and save it to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_image_path": {
                        "type": "string",
                        "description": "The path to the input image file."
                    },
                    "output_image_path": {
                        "type": "string",
                        "description": "The path to the output image file."
                    },
                    "max_width": {
                        "type": "integer",
                        "description": "The maximum width of the output image. Default is None."
                    },
                    "max_height": {
                        "type": "integer",
                        "description": "The maximum height of the output image. Default is None."
                    },
                    "quality": {
                        "type": "integer",
                        "description": "The quality of the output image (1-100). Default is 85."
                    }
                },
                "required": ["input_image_path", "output_image_path", "max_width", "max_height", "quality"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)


def transcribe_audio(input_audio_path: str, output_text_path: str):
    """
    Transcribes audio from an MP3 file and saves the transcription to a text file.
    
    Args:
        input_audio_path (str): The path to the input MP3 file.
        output_text_path (str): The path to the output text file where the transcription will be saved.
    """
    try:

        recognizer = sr.Recognizer()
        with sr.AudioFile(input_audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        with open(output_text_path, 'w') as file:
            file.write(text)
        
        return (f"Transcribed audio from {input_audio_path} and saved to {output_text_path} successfully.")
    except Exception as e:
        print(f"Error transcribing audio: {e}")

tools.append(
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe audio from an MP3 file and save the transcription to a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_audio_path": {
                        "type": "string",
                        "description": "The path to the input MP3 file."
                    },
                    "output_text_path": {
                        "type": "string",
                        "description": "The path to the output text file where the transcription will be saved."
                    }
                },
                "required": ["input_audio_path", "output_text_path"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)


def convert_markdown_to_html(input_file: str, output_file: str):
    """
    Converts a Markdown file to HTML and writes the result to a file.
    
    Args:
        input_file (str): The path to the Markdown file to be converted.
        output_file (str): The path to the file where the HTML output will be written.
    """
    try:

        with open(input_file, 'r') as file:
            markdown_content = file.read()
        
        html_content = markdown.markdown(markdown_content)
        
        with open(output_file, 'w') as file:
            file.write(html_content)
        
        return (f"Converted {input_file} to HTML and saved to {output_file} successfully.")
    except Exception as e:
        print(f"Error converting Markdown to HTML: {e}")

tools.append(
    {
        "type": "function",
        "function": {
            "name": "convert_markdown_to_html",
            "description": "Convert a Markdown file to HTML and write the result to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "The path to the Markdown file to be converted."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the file where the HTML output will be written."
                    }
                },
                "required": ["input_file", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)
    

phase_b=['clone_repo_and_commit','convert_markdown_to_html','transcribe_audio','compress_or_resize_image','scrape_website','run_sql_query','fetch_data_from_api']


META_PROMPT = """
Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - JSON should never be wrapped in code blocks (```) unless explicitly requested.

The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
[If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]
""".strip()

def generate_prompt(task: str):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": META_PROMPT,
            },
            {
                "role": "user",
                "content": f""" You are a prompt engineering assistant. A task description will be provided to you.
    Your goal is to refine and improve this task description to make it more detailed, clear, and effective for AI execution.
    Ensure the output is only the improved prompt text, without any additional commentary.
    
    Task description: "Your goal is to write a python/bash script to perform a given instruction. Make sure the response is only a fenced code block with the code inside.
                The instruction is always programmable and it can either be detailed like "The file `/data/dates.txt` contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to `/data/dates-wednesdays.txt`" or simple like "Convert file.md to HTML".
                Use your best knowledge to implement the same as a program. Remember that the script you provide will be executed in docker container which only has the mentined packages "git, curl, wget, sqlite3, ffmpeg, imagemagick, build-essential, libpq-dev, node v22" and is built atop python:3.12-slim-bookworm.
                Give me only the fenced code block and don't add any further instructions.
                \nInstruction: "{task}" """,
            },
        ],
    }

    completion = handle_openai_api_request(
        url=url,
        data=data,
        method='post',
        json_key='choices.0.message.content'
    )
    return completion


def get_script(task_instruction: str):
    try:
        # prompt=generate_prompt(task_instruction)
        prompt=f"""
Your goal is to write a Python or Bash script to perform the given instruction. The response must be a fenced code block containing only the code.

### Task Details:
- The instruction can be highly detailed (e.g., *"The file `/data/dates.txt` contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to `/data/dates-wednesdays.txt`."*) or simple (e.g., *"Convert `file.md` to HTML"*).
- Implement the instruction using your best knowledge.

### Execution Environment:
- The script will run inside a Docker container with the following pre-installed system packages:  
  `git`, `curl`, `wget`, `sqlite3`, `ffmpeg`, `imagemagick`, `build-essential`, `libpq-dev`, and `node v22`.  

### Response Format:
- **Provide only a fenced code block containing the script**.
- **Do not include any explanations, comments, or additional instructions**.

- The container **includes** the following **Python packages**:  
  - Web frameworks & APIs: `fastapi`, `uvicorn[standard]`, `requests`
  
- **If the script requires any additional Python packages**, it must include an **inline script for `uv` to resolve dependencies**, like this:  

  ```python
  # /// script
  # requires-python = ">=3.13"
  # dependencies = [
  #     "fastapi",
  #     "uvicorn",
  #     "requests",
  #     "db-sqlite3",
  #     "scipy",
  #     "python-dateutil",
  #     "pybase64",
  # ]
  # ///

  import fastapi
  <insert code logic here>
  ```  
(Note: make sure to specify the CORRECT name of the dependency. if unsure, find some other way to implement the same logic.)

### Example Instructions and Expected Outputs:

#### Example 1:
**Instruction:**  
*"The file `/data/dates.txt` contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to `/data/dates-wednesdays.txt`."*

**Expected Output (Python script):**
```python
from datetime import datetime

# Read the file and count Wednesdays
with open('/data/dates.txt', 'r') as file:
    wednesdays = sum(1 for line in file if datetime.strptime(line.strip(), '%Y-%m-%d').weekday() == 2)

# Write the count to the output file
with open('/data/dates-wednesdays.txt', 'w') as file:
    file.write(str(wednesdays))
```

#### Example 2:
**Instruction:**  
*"Convert `file.md` to HTML."*

**Expected Output (Bash script):**
```bash
pandoc file.md -o file.html
```

#### Example 3:
**Instruction:**  
*"Download an image from `https://example.com/image.jpg` and convert it to PNG format."*

**Expected Output (Bash script):**
```bash
wget -O image.jpg https://example.com/image.jpg
convert image.jpg image.png
```

#### Example 4:
**Instruction:**  
*"Fetch the latest 5 commits from a Git repository at `/repo` and save them to `/repo/latest_commits.txt`."*

**Expected Output (Bash script):**
```bash
cd /repo
git log -n 5 --pretty=format:"%H %s" > latest_commits.txt
```

#### Instruction: {task_instruction}"""
        script = handle_openai_api_request(
            url=f"{OPENAI_API_BASE_URL}/chat/completions",
            data={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]},
            json_key="choices.0.message.content"
        )
        script_type = "python" if script.startswith("```python") else "bash"
        script_code = script.split("```")[1].strip()
        if script_code.startswith("python\n"):
            script_code = script_code[len("python\n"):]
        elif script_code.startswith("bash\n"):
            script_code = script_code[len("bash\n"):]
        return( {"script_type": script_type, "script_code": script_code})
    except Exception as e:
        print(f"Exception encountered: {e}")


def handle_diverse_tasks(task_instruction: str):
    print('handling with AI')
    script = get_script(task_instruction)
    script_type = script["script_type"]
    script_code = script["script_code"]
    
    file_extension = "py" if script_type == "python" else "sh"
    def generate_random_filename(extension: str = "txt"):
        """
        Generates a random filename using UUID and the specified extension.
        
        Args:
            extension (str): The file extension (default is "txt").
        
        Returns:
            str: The generated random filename.
        """
        random_filename = f"{uuid.uuid4()}.{extension}"
        return random_filename


    script_file = f"/tmp/{generate_random_filename(file_extension)}"
    
    with open(script_file, 'w') as file:
        file.write(script_code)
    
    if script_type == "python":
        result = subprocess.run(["uv", "run", script_file], capture_output=True, text=True)
    else:
        result = subprocess.run(["bash", script_file], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Script execution failed: {result.stderr}")
        return None
    
    return {"output": result.stdout}


@app.post("/run")
async def run_task(task: str):
    try:
        # Ensure task is not empty
        if not task:
            raise HTTPException(status_code=400, detail="Task parameter is required")

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a function classifier that extracts structured parameters from queries. Note that all the file paths are relative to root folder."},
                {"role": "user", "content": f"{task}"}
            ],
            "tools": tools,
            "tool_choice": "auto"
        }

        function_call = handle_openai_api_request(
            url=f"{OPENAI_API_BASE_URL}/chat/completions",
            data=data,
            json_key="choices.0.message.tool_calls.0.function"
        )

        if not function_call:
            print("No function call returned by OpenAI API")
            return handle_diverse_tasks(task)

        function_name = function_call["name"]
        print(function_name)

        try:
            function_args = json.loads(function_call["arguments"])
            if function_name in phase_b:
                result = handle_diverse_tasks(task)
                if result:
                    return {"result": result}
            
            if function_name in globals():
                function_to_call = globals()[function_name]
                try:
                    result = function_to_call(**function_args)
                    if result:
                        return {"result": result}
                except Exception as e:
                    return handle_diverse_tasks(task)
            else:
                return handle_diverse_tasks(task)
        except HTTPException as http_exc:
            return handle_diverse_tasks(task)
    except Exception as e:
        return handle_diverse_tasks(task)


@app.get("/read")
async def read_file(path: str):
    file_path = Path(path)
    if file_path.exists():
        with open(file_path, 'r') as file:
            content = file.read()
        return PlainTextResponse(content)
    else:
        raise HTTPException(status_code=404, detail=f"File '{path}' not found")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
