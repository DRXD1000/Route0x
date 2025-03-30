import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import ollama
from ollama import Client
import google.generativeai as genai
import traceback
import os

TEMPERATURE = 0
SEED = 1234

class UnifiedLLM:
    def __init__(self, provider, model, ollama_server_url='http://localhost:11434', system_prompt=None):

        self.provider = provider.lower()
        self.model = model
        self.system_prompt = system_prompt
        self.client = None

        if self.provider == 'openai':
            self.api_token = os.getenv("OPENAI_API_KEY")
            if self.api_token is None:
                raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
            openai.api_key = self.api_token

        elif self.provider == 'claude':
            # self.client = Anthropic(api_key=self.api_token)
            raise NotImplementedError("Anthropic support is not yet implemented.")

        elif self.provider == 'google':
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            # raise NotImplementedError("Google support is not yet implemented.")

        elif self.provider == 'ollama':
            self.client = Client(host=ollama_server_url)
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, prompt):
        
        if self.provider == 'openai':
            return self._generate_openai(prompt)
        elif self.provider == 'google':
            return self._generate_google(prompt)
        elif self.provider == 'claude':
            return self._generate_claude(prompt)
        elif self.provider == 'ollama':
            return self._generate_ollama(prompt)



    def _generate_google(self, prompt):
        try:

            generation_config = {
                "temperature": TEMPERATURE,
                "top_p": 1.0,
                "top_k": 0,
                "max_output_tokens": 4000,
                "response_mime_type": "application/json",
            }

            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config,
            )

            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [{"text": self.system_prompt}]
                    }
                ]
            )

            response = chat_session.send_message({"text": prompt})
            response_content = response.text
            return response_content


        except Exception as e:
            traceback.print_exc()


    def _generate_openai(self, prompt):
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=TEMPERATURE,
                seed=SEED,
                response_format = { "type": "json_object" }
            )

            return response.choices[0].message.content
        except Exception as e:
            traceback.print_exc()


    def _generate_claude(self, prompt):
        full_prompt = f"{HUMAN_PROMPT} {self.system_prompt}\n\n{prompt}{AI_PROMPT}"
        response = self.client.completions.create(
            model=self.model,
            prompt=full_prompt,
            max_tokens_to_sample=1000
        )
        return response.completion

    def _generate_ollama(self, prompt):
            response = self.client.chat(model=self.model, messages=[
                    {
                        'role': 'system',
                        'content': self.system_prompt
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    },
             ])

            return response["message"]["content"]
