import json
import boto3
import json_repair
from termcolor import colored
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from lib import frames
from lib import frame_utils



# NOTE: This helper module is kept for historical reference but is not imported
# by the current Module 2 notebooks. They call Bedrock directly via the
# Converse API. Default model updated to the current Module 2 default so anyone
# re-using this helper gets a working model out of the box.
MODEL_ID = 'global.amazon.nova-2-lite-v1:0'
CLAUDE_PRICING = (0.003, 0.015)  # retained for compatibility; not representative of Nova pricing
MODEL_VER = 'bedrock-2023-05-31'

def analyze_conversations(transcript_file):

    messages = []

    # transcript
    transcript_message = make_transcript(transcript_file)
    messages.append(transcript_message)

    # output format?
    messages.append({
        'role': 'assistant',
        'content': 'Got the transcript. What output format?'
    })

    # example output
    example_message = make_conversation_example()
    messages.append(example_message)

    # prefill output
    messages.append({
        'role': 'assistant',
        'content': '{'
    })

    ## system prompt to role play
    system = '''
    You are a media operation assistant who analyses movie transcripts 
    in WebVTT format and suggest chapter points based on the topic 
    changes in the conversations. It is important to read the entire 
    transcripts.
    '''

    ## setting up the model params
    model_params = {
        'anthropic_version': MODEL_VER,
        'max_tokens': 4096,
        'temperature': 0.1,
        'top_p': 0.7,
        'top_k': 20,
        'stop_sequences': ['\n\nHuman:'],
        'system': system,
        'messages': messages
    }

    response['model_params'] = model_params
    
    try:
        response = inference(model_params)
    except Exception as e:
        print(colored(f"ERR: inference: {str(e)}\n RETRY...", 'red'))
        response = inference(model_params)
    return response

def get_contextual_information(images, text, iab_definitions):
    task_all = 'You are asked to provide the following information: a detail description to describe the scene, identify the most relevant IAB taxonomy, GARM, sentiment, and brands and logos that may appear in the scene, and five most relevant tags from the scene.'
    task_iab_only = 'You are asked to identify the most relevant IAB taxonomy.'
    system = 'You are a media operation engineer. Your job is to review a portion of a video content presented in a sequence of consecutive images. Each image also contains a sequence of frames presented in a 4x7 grid reading from left to right and then from top to bottom. You may also optionally be given the conversation of the scene that helps you to understand the context. {0} It is important to return the results in JSON format and also includes a confidence score from 0 to 100. Skip any explanation.';

    messages = []
 
    # adding sequences of composite images to the prompt.  Limit is 20.
    message_images = make_image_message(images[:19])
    messages.append(message_images)

    # adding the conversation to the prompt
    messages.append({
        'role': 'assistant',
        'content': 'Got the images. Do you have the conversation of the scene?'
    })
    message_conversation = make_conversation_message(text)
    messages.append(message_conversation)

    # other information
    messages.append({
        'role': 'assistant',
        'content': 'OK. Do you have other information to provdie?'
    })

    other_information = []
    ## iab taxonomy
    iab_list = make_iab_taxonomoies(iab_definitions['tier1'])
    other_information.append(iab_list)

    ## GARM
    garm_list = make_garm_taxonomoies()
    other_information.append(garm_list)

    ## Sentiment
    sentiment_list = make_sentiments()
    other_information.append(sentiment_list)

    messages.append({
        'role': 'user',
        'content': other_information
    })

    # output format
    messages.append({
        'role': 'assistant',
        'content': 'OK. What output format?'
    })
    output_format = make_output_example()
    messages.append(output_format)

    # prefill '{'
    messages.append({
        'role': 'assistant',
        'content': '{'
    })
    
    model_params = {
        'anthropic_version': MODEL_VER,
        'max_tokens': 4096,
        'temperature': 0.1,
        'top_p': 0.7,
        'top_k': 20,
        'stop_sequences': ['\n\nHuman:'],
        'system': system.format(task_all),
        'messages': messages
    }

    try:
        response = inference(model_params)
    except Exception as e:
        print(colored(f"ERR: inference: {str(e)}\n RETRY...", 'red'))
        response = inference(model_params)

    return response

def display_prompt(model_params):
    print (f'MODEL_ID: {MODEL_ID}\n')
    print (f'System Prompt:\n\n{model_params["system"]}')
    print (f'Messages:\n\n')
    for message in model_params['messages']:
        print (json.dumps(message))

    print('\n')

    return
    
def inference(model_params):
    model_id = MODEL_ID
    accept = 'application/json'
    content_type = 'application/json'

    bedrock_runtime_client = boto3.client(service_name='bedrock-runtime')

    response = bedrock_runtime_client.invoke_model(
        body=json.dumps(model_params),
        modelId=model_id,
        accept=accept,
        contentType=content_type
    )

    response_body = json.loads(response.get('body').read())

    # patch the json string output with '{' and parse it
    response_content = response_body['content'][0]['text']
    if response_content[0] != '{':
        response_content = '{' + response_content

    try:
        response_content = json.loads(response_content)
    except Exception as e:
        print(colored("Malformed JSON response. Try to repair it...", 'red'))
        try:
            response_content = json_repair.loads(response_content, strict=False)
        except Exception as e:
            print(colored("Failed to repair the JSON response...", 'red'))
            print(colored(response_content, 'red'))
            raise e

    response_body['content'][0]['json'] = response_content

    return response_body

def display_conversation_cost(response):
    # us-east-1 pricing
    input_per_1k, output_per_1k = CLAUDE_PRICING

    input_tokens = response['usage']['input_tokens']
    output_tokens = response['usage']['output_tokens']

    conversation_cost = (
        input_per_1k * input_tokens +
        output_per_1k * output_tokens
    ) / 1000

    print('\n')
    print('========================================================================')
    print('Estimated cost:', colored(f"${conversation_cost}", 'green'), f"in us-east-1 region with {colored(input_tokens, 'green')} input tokens and {colored(output_tokens, 'green')} output tokens.")
    print('========================================================================')

    return {
        'input_per_1k': input_per_1k,
        'output_per_1k': output_per_1k,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'estimated_cost': conversation_cost,
    }

def display_contextual_cost(usage):
    # us-east-1 pricing
    input_per_1k, output_per_1k = CLAUDE_PRICING

    input_tokens = usage['input_tokens']
    output_tokens = usage['output_tokens']

    contextual_cost = (
        input_per_1k * input_tokens +
        output_per_1k * output_tokens
    ) / 1000

    print('\n')
    print('========================================================================')
    print('Estimated cost:', colored(f"${round(contextual_cost, 4)}", 'green'), f"in us-east-1 region with {colored(input_tokens, 'green')} input tokens and {colored(output_tokens, 'green')} output tokens.")
    print('========================================================================')

    return {
        'input_per_1k': input_per_1k,
        'output_per_1k': output_per_1k,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'estimated_cost': contextual_cost,
    }


def make_image_message(composite_images):
    # adding the composite image sequences
    image_contents = [{
        'type': 'text',
        'text': 'Here are {0} images containing frame sequence that describes a scene.'.format(len(composite_images))
    }]

    open_images = []
    for image in composite_images:
        with open(image['file'], "rb") as image_file:
            image_data = image_file.read()
            open_images.append(image_file)
        image_pil = Image.open(BytesIO(image_data))
        bas64_image = frame_utils.image_to_base64(image_pil)
        image_contents.append({
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': 'image/jpeg',
                'data': bas64_image
            }
        })

    # close the images
    for image in open_images:
        image.close()

    return {
        'role': 'user',
        'content': image_contents
    }

def make_conversation_message(text):
    message = {
        'role': 'user',
        'content': 'No conversation.'
    }

    if text:
        message['content'] = 'Here is the conversation of the scene in <conversation> tag.\n<conversation>\n{0}\n</conversation>\n'.format(text)

    return message

def make_iab_taxonomoies(iab_list):
    iab = [item['name'] for item in iab_list]
    iab.append('None')

    return {
        'type': 'text',
        'text': 'Here is a list of IAB Taxonomies in <iab> tag:\n<iab>\n${0}\n</iab>\nOnly answer the IAB taxonomy from this list.'.format('\n'.join(iab))
    }

def make_garm_taxonomoies():
    garm = [
        'Adult & Explicit Sexual Content',
        'Arms & Ammunition',
        'Crime & Harmful acts to individuals and Society, Human Right Violations',
        'Death, Injury or Military Conflict',
        'Online piracy',
        'Hate speech & acts of aggression',
        'Obscenity and Profanity, including language, gestures, and explicitly gory, graphic or repulsive content intended to shock and disgust',
        'Illegal Drugs, Tobacco, ecigarettes, Vaping, or Alcohol',
        'Spam or Harmful Content',
        'Terrorism',
        'Debated Sensitive Social Issue',
        'None',
    ]

    return {
        'type': 'text',
        'text': 'Here is a list of GARM Taxonomies in <garm> tag:\n<garm>\n{0}\n</garm>\nOnly answer the GARM taxonomy from this list.'.format('\n'.join(garm))
    }

def make_sentiments():
    sentiments = ['Positive', 'Neutral', 'Negative', 'None']

    return {
        'type': 'text',
        'text': 'Here is a list of Sentiments in <sentiment> tag:\n<sentiment>\n{0}\n</sentiment>\nOnly answer the sentiment from this list.'.format('\n'.join(sentiments))
    }

def make_output_example():
    example = {
        'description': {
            'text': 'The scene describes...',
            'score': 98
        },
        'sentiment': {
            'text': 'Positive',
            'score': 90
        },
        'iab_taxonomy': {
            'text': 'Station Wagon',
            'score': 80
        },
        'garm_taxonomy': {
            'text': 'Online piracy',
            'score': 90
        },
        'brands_and_logos': [
            {
                'text': 'Amazon',
                'score': 95
            },
            {
                'text': 'Nike',
                'score': 85
            }
        ],
        'relevant_tags': [
            {
                'text': 'BMW',
                'score': 95
            }
        ]            
    }
    
    return {
        'role': 'user',
        'content': 'Return JSON format. An example of the output:\n{0}\n'.format(json.dumps(example))
    }

