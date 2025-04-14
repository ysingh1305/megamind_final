# MegaMind

MegaMind is an AI-powered learning assistant that helps students better comprehend and retain lecture material. Winner of a track at HackRU Spring 2025.

## What It Does

MegaMind is designed for students who want to extract more value from their lectures. By uploading audio or video recordings of lectures, the app creates structured notes, quizzes, and progress visualizations to reinforce understanding.

## How It Works

1. **Input**  
   Users upload lecture audio or video to the app.

2. **Transcription**  
   We use **Whisper AI** to transcribe lecture audio into text.

3. **Natural Language Processing**  
   Using **spaCy**, we extract key terms and phrases from the transcription.

4. **Note & Quiz Generation**  
   Leveraging **LangChain** and **OpenAI’s GPT API**, we turn the extracted content into:
   - Structured notes
   - Quiz questions for self-assessment

5. **Visualization**  
   With **Matplotlib**, we generate a dashboard of visualizations that reflect:
   - Quiz performance
   - Topic comprehension
   - Progress over time
