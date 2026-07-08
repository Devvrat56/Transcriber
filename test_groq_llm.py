import os
from dotenv import load_dotenv
load_dotenv()
from llm import MedicalLLM

llm = MedicalLLM(model="llama-3.1-8b-instant")
transcript = "Patient came with severe chest pain. Blood pressure is 120/80. Prescribed aspirin 81mg."
summary = llm.generate_doctor_summary(transcript, {})
print("llama-3.1-8b-instant SUMMARY:")
print(summary)

llm_70b = MedicalLLM(model="llama-3.3-70b-versatile")
summary_70b = llm_70b.generate_doctor_summary(transcript, {})
print("\nllama-3.3-70b-versatile SUMMARY:")
print(summary_70b)
