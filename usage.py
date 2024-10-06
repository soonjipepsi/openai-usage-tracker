import requests
import datetime
import pandas as pd
import numpy as np
import json

# 2024-10-07 Cost Information
cost_rates = {
    'gpt-4o-2024-05-13': {'input': 0.00500, 'output': 0.01500},
    'gpt-4o-2024-08-06': {'input': 0.00250, 'output': 0.01000},
    'gpt-4o': {'input': 0.00250, 'output': 0.01000},
    'gpt-4': {'input': 0.03000, 'output': 0.06000},
    'gpt-4-turbo': {'input': 0.01000, 'output': 0.03000},
    'gpt-4o-mini-2024-07-18': {'input': 0.000150, 'output': 0.000600},
    'gpt-4o-mini': {'input': 0.000150, 'output': 0.000600},
    'gpt-3.5-turbo': {'input': 0.003000, 'output': 0.006000},
    'text-embedding-3-small': {'input': 0.000020},
    'text-embedding-3-large': {'input': 0.000130},
    'ada-v2': {'input': 0.000100},
    'dall-e-3-standard': {'input': 0.040},   # per image
    'dall-e-3-hd': {'input': 0.080},         # per image
    'whisper': {'input': 0.006 / 60},        # per second
    'tts': {'input': 0.015 / 1000},          # per character
    'tts-hd': {'input': 0.030 / 1000},       # per character
    'o1-preview': {'input': 0.01500, 'output': 0.06000},
    'o1-mini': {'input': 0.00300, 'output': 0.01200},
    'gpt-4o-realtime-preview': {'input': 0.00500, 'output': 0.02000}
}

# Simplify model name function
def simplify_model_name(model_name):
    if 'gpt-4' in model_name and 'turbo' in model_name:
        return 'gpt-4-turbo'
    elif 'gpt-4o' in model_name and 'mini' in model_name and '2024-07-18' in model_name:
        return 'gpt-4o-mini-2024-07-18'
    elif 'gpt-4o' in model_name and 'mini' in model_name:
        return 'gpt-4o-mini'
    elif 'gpt-4o' in model_name and '2024-08-06' in model_name:
        return 'gpt-4o-2024-08-06'
    elif 'gpt-4o' in model_name and '2024-05-13' in model_name:
        return 'gpt-4o-2024-05-13'
    elif 'gpt-4o' in model_name:
        return 'gpt-4o'
    elif 'gpt-4' in model_name:
        return 'gpt-4'
    elif 'gpt-3.5' in model_name:
        return 'gpt-3.5-turbo'
    elif 'o1-preview' in model_name:
        return 'o1-preview'
    elif 'o1-mini' in model_name:
        return 'o1-mini'
    elif 'text-embedding-3-small' in model_name:
        return 'text-embedding-3-small'
    elif 'text-embedding-3-large' in model_name:
        return 'text-embedding-3-large'
    elif 'ada-v2' in model_name:
        return 'ada-v2'
    elif 'dall-e-3' in model_name and 'hd' in model_name.lower():
        return 'dall-e-3-hd'
    elif 'dall-e-3' in model_name:
        return 'dall-e-3-standard'
    elif 'whisper' in model_name:
        return 'whisper'
    elif 'tts-hd' in model_name:
        return 'tts-hd'
    elif 'tts' in model_name:
        return 'tts'
    elif 'gpt-4o-realtime-preview' in model_name:
        return 'gpt-4o-realtime-preview'
    else:
        return 'other'

# Function to calculate the cost
def calculate_cost(row):
    model = row['simplified_model']
    rates = cost_rates.get(model, {})
    if model in [
        'gpt-4', 'gpt-4-turbo',
        'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06', 'gpt-4o',
        'gpt-4o-mini-2024-07-18', 'gpt-4o-mini',
        'gpt-3.5-turbo', 'o1-preview', 'o1-mini',
        'gpt-4o-realtime-preview'
    ]:
        input_cost = (row['total_context_tokens'] / 1000) * rates.get('input', 0)
        output_cost = (row['total_generated_tokens'] / 1000) * rates.get('output', 0)
        return input_cost + output_cost
    elif model in ['text-embedding-3-small', 'text-embedding-3-large', 'ada-v2']:
        total_tokens = row['total_context_tokens'] + row['total_generated_tokens']
        cost = (total_tokens / 1000) * rates.get('input', 0)
        return cost
    elif model in ['dall-e-3-standard', 'dall-e-3-hd']:
        cost = row['total_requests'] * rates.get('input', 0)
        return cost
    elif model == 'whisper':
        cost = row['total_seconds'] * rates.get('input', 0)
        return cost
    elif model in ['tts', 'tts-hd']:
        cost = row['total_characters'] * rates.get('input', 0)
        return cost
    else:
        return 0


# Case 1: Fetch data from API using input_date
def case1(input_date):

    # Load API key from config.json
    with open('config.json') as config_file:
        config = json.load(config_file)
        API_KEY = config['API_KEY']
        
    if API_KEY == 'YOUR_API_KEY' :
        print("API key is missing. Please provide a valid API key in the config.json file.")
        return 
    # Set API headers
    headers = {'Authorization': f'Bearer {API_KEY}'}
    
    # API endpoint
    url = 'https://api.openai.com/v1/usage'
    
    # API request parameters
    params = {'date': input_date}
    
    # API request and response
    response = requests.get(url, headers=headers, params=params)
    
    # Extract data from the response
    usage_data = response.json().get('data', [])
    
    # Convert to DataFrame
    data = pd.DataFrame(usage_data)
    
    # Check if data is empty
    if data.empty:
        print("No usage data found for the specified date.")
        return
    
    # Process the data
    data['date'] = pd.to_datetime(data['aggregation_timestamp'], unit='s')
    
    # Select necessary columns and rename them
    filtered_data = data[['snapshot_id', 'n_context_tokens_total', 'n_generated_tokens_total',
                          'n_requests', 'date']].copy()
    filtered_data.rename(columns={'snapshot_id': 'model', 'n_requests': 'num_requests'}, inplace=True)
    
    # Add num_characters and num_seconds columns (if missing)
    filtered_data['num_characters'] = 0
    filtered_data['num_seconds'] = 0
    
    # Simplify model names
    filtered_data['simplified_model'] = filtered_data['model'].apply(simplify_model_name)
    
    # Group and aggregate the data
    grouped_data = filtered_data.groupby('simplified_model').agg(
        total_requests=pd.NamedAgg(column='num_requests', aggfunc='sum'),
        total_context_tokens=pd.NamedAgg(column='n_context_tokens_total', aggfunc='sum'),
        total_generated_tokens=pd.NamedAgg(column='n_generated_tokens_total', aggfunc='sum'),
        total_characters=pd.NamedAgg(column='num_characters', aggfunc='sum'),
        total_seconds=pd.NamedAgg(column='num_seconds', aggfunc='sum'),
        start_date=pd.NamedAgg(column='date', aggfunc='min'),
        end_date=pd.NamedAgg(column='date', aggfunc='max')
    ).reset_index()
    
    # Calculate costs
    grouped_data['total_cost'] = grouped_data.apply(calculate_cost, axis=1)
    
    # Overall statistics
    overall_start_date = filtered_data['date'].min()
    overall_end_date = filtered_data['date'].max()
    total_days = (overall_end_date - overall_start_date).days + 1  
    
    total_calls = grouped_data['total_requests'].sum()
    total_cost = grouped_data['total_cost'].sum()
    daily_average_calls = total_calls / total_days
    daily_average_cost = total_cost / total_days
    
    # Calculate daily averages per model
    grouped_data['model_total_days'] = (grouped_data['end_date'] - grouped_data['start_date']).dt.days + 1
    grouped_data['daily_average_calls'] = grouped_data['total_requests'] / grouped_data['model_total_days']
    grouped_data['daily_average_cost'] = grouped_data['total_cost'] / grouped_data['model_total_days']
    
    # Print results
    print(f"Total calls: {total_calls}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Daily average calls: {daily_average_calls:.2f}")
    print(f"Daily average cost: ${daily_average_cost:.2f}")
    
    print("\nUsage by model:")
    print(grouped_data[['simplified_model', 'total_requests', 'total_cost', 'daily_average_calls', 'daily_average_cost']])

# Case 2: Read data from CSV file
def case2(file_path):
    # Read CSV file
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # Convert timestamp to date format
    data['date'] = pd.to_datetime(data['timestamp'], unit='s')
    
    # Select and copy the necessary columns
    filtered_data = data[['model', 'n_context_tokens_total', 'n_generated_tokens_total',
                          'num_requests', 'num_characters', 'num_seconds', 'date']].copy()
    
    # Simplify model names
    filtered_data['simplified_model'] = filtered_data['model'].apply(simplify_model_name)
    
    # Group and aggregate the data
    grouped_data = filtered_data.groupby('simplified_model').agg(
        total_requests=pd.NamedAgg(column='num_requests', aggfunc='sum'),
        total_context_tokens=pd.NamedAgg(column='n_context_tokens_total', aggfunc='sum'),
        total_generated_tokens=pd.NamedAgg(column='n_generated_tokens_total', aggfunc='sum'),
        total_characters=pd.NamedAgg(column='num_characters', aggfunc='sum'),
        total_seconds=pd.NamedAgg(column='num_seconds', aggfunc='sum'),
        start_date=pd.NamedAgg(column='date', aggfunc='min'),
        end_date=pd.NamedAgg(column='date', aggfunc='max')
    ).reset_index()
    
    # Calculate costs
    grouped_data['total_cost'] = grouped_data.apply(calculate_cost, axis=1)
    
    # Overall statistics
    overall_start_date = filtered_data['date'].min()
    overall_end_date = filtered_data['date'].max()
    total_days = (overall_end_date - overall_start_date).days + 1  
    
    total_calls = grouped_data['total_requests'].sum()
    total_cost = grouped_data['total_cost'].sum()
    daily_average_calls = total_calls / total_days
    daily_average_cost = total_cost / total_days
    
    # Calculate daily averages per model
    grouped_data['model_total_days'] = (grouped_data['end_date'] - grouped_data['start_date']).dt.days + 1
    grouped_data['daily_average_calls'] = grouped_data['total_requests'] / grouped_data['model_total_days']
    grouped_data['daily_average_cost'] = grouped_data['total_cost'] / grouped_data['model_total_days']
    
    # Print results
    print(f"Total calls: {total_calls}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Daily average calls: {daily_average_calls:.2f}")
    print(f"Daily average cost: ${daily_average_cost:.2f}")
    
    print("\nUsage by model:")
    print(grouped_data[['simplified_model', 'total_requests', 'total_cost', 'daily_average_calls', 'daily_average_cost']])

# Main execution
if __name__ == "__main__":
    # Choose case
    choice = input("Choose a case (1: Use input_date, 2: Use CSV file): ")
    if choice == '1':
        input_date = input("Enter a date (YYYY-MM-DD): ")
        case1(input_date)
    elif choice == '2':
        file_path = input("Enter the path to the CSV file: ")
        case2(file_path)
    else:
        print("Invalid choice. Exiting.")
