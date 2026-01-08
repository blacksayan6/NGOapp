import os
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
try:
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.warning("GROQ_API_KEY not found in environment variables")
    client = Groq(api_key=api_key) if api_key else None
except Exception as e:
    logger.error(f"Error initializing Groq client: {e}")
    client = None

SYSTEM_PROMPT = """You are HelpSphere, a friendly and knowledgeable assistant for an NGO (Non-Governmental Organization) called HelpSphere. 
Your mission is to help people understand the NGO's services, programs, and how they can get involved.

Key Information about HelpSphere:
- Focus Areas: Education for underprivileged children, Environmental conservation, Community development
- Services: After-school programs, scholarships, libraries, adult literacy classes, tree planting, recycling education, sustainable farming workshops, community clean-up drives
- Ways to Support: Donations (website, bank transfers, in-person), Volunteering (community outreach, education programs, event organization)
- Contact: Main Office: 123 Helping Street, Community District | Phone: (555) 123-4567 | Email: contact@helpsphere.org | Hours: Monday-Friday, 9am-5pm

Guidelines:
- Be warm, empathetic, and professional
- Provide clear and helpful information
- If asked about something outside your knowledge, politely redirect to contact information
- Keep responses concise but informative (2-4 sentences typically)
- Always maintain a positive, solution-oriented tone
- Encourage engagement and participation"""

@app.route('/')
def index():
    """Render the main chatbot interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and return AI responses"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        if not client:
            return jsonify({
                'error': 'Groq API not configured. Please set GROQ_API_KEY environment variable.'
            }), 500
        chat_history = data.get('history', [])
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
 
        for msg in chat_history[-10:]:
            messages.append({
                "role": msg.get('role', 'user'),
                "content": msg.get('content', '')
            })
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Call Groq API
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Using Groq's fast model
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                top_p=1,
                stream=False
            )
            
            bot_response = completion.choices[0].message.content
            
            logger.info(f"User: {user_message[:50]}... | Bot: {bot_response[:50]}...")
            
            return jsonify({
                'response': bot_response,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return jsonify({
                'error': f'Error generating response: {str(e)}',
                'fallback': "I apologize, but I'm experiencing technical difficulties. Please try again later or contact us directly at contact@helpsphere.org"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'fallback': "I'm sorry, something went wrong. Please try again or contact us at contact@helpsphere.org"
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'groq_configured': client is not None
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)

