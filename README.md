# LiveKit Food Ordering Bot

A food ordering bot built with LiveKit, Twilio, and Pipecat.

## Deployment to Railway

1. Create a new project on [Railway](https://railway.app)
2. Connect your GitHub repository
3. Set up the following environment variables in Railway:
   - TWILIO_ACCOUNT_SID
   - TWILIO_AUTH_TOKEN
   - LIVEKIT_API_KEY
   - LIVEKIT_API_SECRET
   (Add any other environment variables from your .env file)
4. Deploy the application

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your environment variables

3. Run the application:
```bash
python food_ordering_livekit.py
```
