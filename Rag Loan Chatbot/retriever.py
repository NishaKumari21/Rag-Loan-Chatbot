import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class Retriever:
    def __init__(self, csv_path='data/Training Dataset.csv'):
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
        self.embeddings = None

    def prepare_documents(self):
        self.documents = [
            f"Loan ID: {row['Loan_ID']}, Gender: {row['Gender']}, Married: {row['Married']}, "
            f"Education: {row['Education']}, ApplicantIncome: {row['ApplicantIncome']}, "
            f"LoanAmount: {row['LoanAmount']}, Loan_Status: {row['Loan_Status']}"
            for _, row in self.df.iterrows()
        ]
        return self.documents

    def create_index(self):
        self.embeddings = self.model.encode(self.documents)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

    def retrieve_context(self, query, k=3):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), k)
        return "\n".join([self.documents[i] for i in indices[0]])

