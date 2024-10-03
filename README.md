# OpenAI API Usage Tracker

**Version**: 0.0.1 
**Release Date**: October 2, 2024

This project is a Python script designed to track usage and cost data from the OpenAI API. It fetches the data either directly from the API or from a CSV file, processes it, and calculates the total usage cost based on the model and token consumption.

## CSV Data Export

You can export usage data as a CSV file directly from the [OpenAI Platform usage page](https://platform.openai.com/usage). The script also supports setting a custom date range for more specific data analysis.

## Features
- Fetches usage data from the OpenAI API by date.
- Reads usage data from a CSV file for offline processing.
- Calculates the cost for different models, including GPT-4, GPT-3.5, DALL-E 3, Whisper, and text-to-speech (TTS).
- Provides a summary of the total usage, total cost, and daily averages.

## Requirements
To run this project, you need the following:

- Python 3.x
- `requests` library
- `pandas` library
- `numpy` library

You can install the required libraries using pip:

```bash
pip install requests pandas numpy
