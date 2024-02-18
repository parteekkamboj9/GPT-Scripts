from datetime import datetime
import mysql.connector
from mysql.connector import pooling
import os
import re
from flask import Flask, request, jsonify
import openai
from flask import Flask, request, jsonify, abort
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import requests
from langchain.chains import LLMChain

app = Flask(__name__)

# OpenAI API key
api_key = "enter-your-openai-key-here"
os.environ['OPENAI_API_KEY'] = api_key
openai.api_key = api_key

# Pixel API key
pixel_api_key = 'enter-your-pixel-api-key-here'

# MySQL database connection configuration
host = "localhost"
user = "root"
password = "root"
database = "enter-your-db-name-here"


# Function to connect to the MySQL database
# def connect_mysql_db():
#     try:
#         return pooling.MySQLConnectionPool(
#             pool_name="my_pool",
#             pool_size=10,
#             user=user,
#             password=password,
#             host=host,
#             database=database
#         ).get_connection()
#     except mysql.connector.Error as e:
#         return jsonify(error=f"Database connection error: {e}", status_code=500), 500

# connect mysql 
def connect_mysql_db():
    try:
        db_connection = pooling.MySQLConnectionPool(
            pool_name="my_pool",
            pool_size=10,
            user=user,
            password=password,
            host=host,
            database=database
        ).get_connection()
        return db_connection  # Return the connection object, not a tuple
    except mysql.connector.Error as e:
        return jsonify(error=f"Database connection error: {e}", status_code=500)


# To get currunt  datetime
def date_now():
    import pytz
    from datetime import datetime
    Today = datetime.utcnow().replace(tzinfo=pytz.utc)
    return Today.date()


# Search images according to query & pages
def search_pexels_images(query, per_page=1):
    headers = {
        'Authorization': pixel_api_key
    }

    url = f'https://api.pexels.com/v1/search?query={query}&per_page={per_page}'

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError('Request to Pexels API failed, try again.')
        # return {'error': 'Request to Pexels API failed'}


# Create dict from str output using regex pattern
def create_dict_from_output(output, add_image):
    slides = []

    # Define a regular expression pattern to capture the slide information
    slide_pattern = re.compile(r'SLIDE NO: (\d+)\nTITLE: (.+?)\nDESCRIPTION:(.+?)(?=\nSLIDE NO:|$)', re.DOTALL)

    for count, match in enumerate(slide_pattern.finditer(output)):
        image_src = None
        slide_number = match.group(1)
        title = match.group(2)
        description = match.group(3).strip()  # Strip leading/trailing whitespace

        if count in [0, 1, 2] and add_image:
            try:
                # Call the search_pexels_images function to get image source
                image_json = search_pexels_images(query=title)
                image_src = image_json['photos'][0]['src']['original']
            except Exception as e:
                # Handle any exceptions that may occur during image retrieval
                print(f"Error while fetching image: {str(e)}")

        slide_info = {
            "Slide Number": slide_number,
            "Title": title,
            "Description": description,
            "ImageSrc": image_src
        }
        slides.append(slide_info)

    return slides


@app.route("/user", methods=['GET'])
def get_user():
    user_email = request.args.get('user_email')
    if user_email:
        try:
            db_connection = connect_mysql_db()
            cursor = db_connection.cursor(dictionary=True)

            query = "SELECT * FROM User WHERE email = %s"
            cursor.execute(query, (user_email,))

            user_data = cursor.fetchone()

            cursor.close()
            db_connection.close()

            if user_data:
                return jsonify(user_data)
            else:
                return jsonify(error="User not found", status_code=404), 404
        except mysql.connector.Error as e:
            return jsonify(error=f"Database error: {e}", status_code=500), 500
    else:
        return jsonify(error="user_email parameter is required", status_code=400), 400


@app.route("/ask-text", methods=['POST'])
def ask_question_text():
    from datetime import datetime
    data = request.get_json()
    user_email = data.get('user_email')
    question = data.get('question')
    no_of_slide = data.get('no_of_slide')
    type_of_slide = data.get('type_of_slide')
    add_image = data.get('add_image')

    try:
        db_connection = connect_mysql_db()
        cursor = db_connection.cursor(dictionary=True)

        query = "SELECT * FROM User WHERE email = %s"
        cursor.execute(query, (user_email,))

        user_data = cursor.fetchone()

        if not user_data:
            cursor.close()
            db_connection.close()
            abort(404, description="User not found")

        available_quota = user_data['available_quota']
        available_character = user_data['available_character']
        expiry_date = user_data['expiry_date']

        # if available_quota <= 0 or available_character <= 0:
        if available_quota <= 0 or available_character <= 0 or expiry_date.date() <= date_now():
            abort(422, description="You exceed your quota/plan. Please upgrade your plan")

        prompt_template = f"""Create a {no_of_slide}-point {type_of_slide} presentation on 
{question}.

Use this format for slide every slide including 
SLIDE NO:
TITLE:
DESCRIPTION:


In every slide Tittle should be under 50 characters and the descriptions should be under 750 characters and description always in paragraphs not in points.

In 4 slide title should be under 50 characters and description in 4 points and every point under 100 characters 


Please follow this rule strictly
"""

        # Create the LLMChain instance
        llm_chain = LLMChain(llm=ChatOpenAI(temperature=0.4), prompt=PromptTemplate.from_template(prompt_template))

        # Generate output
        output = llm_chain({"question": question})
        output_text = output['text']
        # print(output_text, "--------output_text--------------------")
        response = create_dict_from_output(output=output_text, add_image=add_image)

        if response:
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            available_quota -= 1
            available_character -= len(question)

            if available_character <= 0:
                available_character = 0
            print(created_at, "--------------created_at")
            cursor = db_connection.cursor(dictionary=True)

            query = "UPDATE User SET available_quota = %s, available_character = %s WHERE email = %s"
            cursor.execute(query, (available_quota, available_character, user_email))
            db_connection.commit()
            insert_query = "INSERT INTO user_history (prompt, created_at, email) VALUES (%s, %s, %s)"
            cursor.execute(insert_query, (question[:100], created_at, user_email))
            db_connection.commit()

        cursor.close()
        db_connection.close()
        # print(response)

        return jsonify(response)

    except mysql.connector.Error as e:
        abort(500, description=f"Database error: {e}")
    except Exception as e:
        abort(500, description=str(e))


@app.route("/ask-topic", methods=['POST'])
def ask_question_topic():
    from datetime import datetime
    data = request.get_json()
    user_email = data.get('user_email')
    question = data.get('question')
    no_of_slide = data.get('no_of_slide')
    type_of_slide = data.get('type_of_slide')
    add_image = data.get('add_image')

    try:
        db_connection = connect_mysql_db()
        cursor = db_connection.cursor(dictionary=True)

        query = "SELECT * FROM User WHERE email = %s"
        cursor.execute(query, (user_email,))

        user_data = cursor.fetchone()

        if not user_data:
            cursor.close()
            db_connection.close()
            abort(404, description="User not found")

        available_quota = user_data['available_quota']
        available_character = user_data['available_character']
        expiry_date = user_data['expiry_date']

        # if available_quota <= 0 or available_character <= 0:
        if available_quota <= 0 or available_character <= 0 or expiry_date.date() <= date_now():
            abort(422, description="You exceed your quota/plan. Please upgrade your plan")

        editiona_info = ''"
        if question:
            editiona_info = "with editional information " + question + ' '
        prompt_template = f"""Create a {no_of_slide} point informational presentation on
{type_of_slide} {editiona_info}Expand on each of the subtopics. You can consider elaborating on the key ideas, offering supporting examples and explaining any details that you think would enhance the audience's understanding on the topic.


Use this format for slide every slide including 
SLIDE NO:
TITLE:
DESCRIPTION:


In slide, 1,2,3,5,6,7,8,9,10 Tittle should be under 50 characters and the descriptions should be 750 characters and description always in paragraphs not in points.

In 4 slide title should be under 50 characters and description in 4 points and every point under 100 characters 


Please follow this rule strickly
"""

        # Create the LLMChain instance
        llm_chain = LLMChain(llm=ChatOpenAI(temperature=0.4), prompt=PromptTemplate.from_template(prompt_template))

        # Generate output
        output = llm_chain({"question": question})
        output_text = output['text']
        # print(output_text, "--------output_text--------------------")
        response = create_dict_from_output(output=output_text, add_image=add_image)

        if response:
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            available_quota -= 1
            available_character -= len(question)

            if available_character <= 0:
                available_character = 0

            print(created_at, "--------------created_at")
            cursor = db_connection.cursor(dictionary=True)

            query = "UPDATE User SET available_quota = %s, available_character = %s WHERE email = %s"
            cursor.execute(query, (available_quota, available_character, user_email))
            db_connection.commit()
            insert_query = "INSERT INTO user_history (prompt, created_at, email) VALUES (%s, %s, %s)"
            cursor.execute(insert_query, (question[:100], created_at, user_email))
            db_connection.commit()

        cursor.close()
        db_connection.close()
        # print(response)

        return jsonify(response)

    except mysql.connector.Error as e:
        abort(500, description=f"Database error: {e}")
    except Exception as e:
        abort(500, description=str(e))


if __name__ == '__main__':
    # app.run()
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)
