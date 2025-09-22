#!/usr/bin/env python3
"""
Cultural Heritage Chatbot
A simple chatbot that answers questions about cultural heritage sites and monuments.
"""

import json
import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

class CulturalHeritageChatbot:
    def __init__(self, knowledge_base_path: str = "data/heritage_docs.json"):
        """Initialize the chatbot with a knowledge base."""
        self.documents = []
        self.tfidf_vectors = []
        self.vocabulary = set()
        
        # Load knowledge base
        try:
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"Loaded {len(self.documents)} documents from knowledge base.")
            self._build_tfidf_index()
        except FileNotFoundError:
            print(f"Error: Could not find knowledge base at {knowledge_base_path}")
            print("Please make sure the file exists and try again.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {knowledge_base_path}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by converting to lowercase and extracting words."""
        # Convert to lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Simple stopword removal
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their', 'them'
        }
        
        return [word for word in words if word not in stopwords and len(word) > 2]
    
    def _build_tfidf_index(self):
        """Build TF-IDF index for all documents."""
        if not self.documents:
            return
        
        # Preprocess all documents and build vocabulary
        processed_docs = []
        for doc in self.documents:
            # Combine title, content, and keywords for indexing
            text = f"{doc['title']} {doc['content']} {' '.join(doc.get('keywords', []))}"
            words = self._preprocess_text(text)
            processed_docs.append(words)
            self.vocabulary.update(words)
        
        self.vocabulary = sorted(list(self.vocabulary))
        
        # Calculate TF-IDF for each document
        doc_count = len(processed_docs)
        word_doc_count = defaultdict(int)
        
        # Count document frequency for each word
        for words in processed_docs:
            unique_words = set(words)
            for word in unique_words:
                word_doc_count[word] += 1
        
        # Calculate TF-IDF vectors
        for words in processed_docs:
            word_count = Counter(words)
            doc_length = len(words)
            
            tfidf_vector = {}
            for word in self.vocabulary:
                tf = word_count[word] / doc_length if doc_length > 0 else 0
                idf = math.log(doc_count / word_doc_count[word]) if word_doc_count[word] > 0 else 0
                tfidf_vector[word] = tf * idf
            
            self.tfidf_vectors.append(tfidf_vector)
    
    def _calculate_cosine_similarity(self, query_vector: Dict[str, float], doc_vector: Dict[str, float]) -> float:
        """Calculate cosine similarity between query and document vectors."""
        # Calculate dot product
        dot_product = sum(query_vector.get(word, 0) * doc_vector.get(word, 0) for word in self.vocabulary)
        
        # Calculate magnitudes
        query_magnitude = math.sqrt(sum(val ** 2 for val in query_vector.values()))
        doc_magnitude = math.sqrt(sum(val ** 2 for val in doc_vector.values()))
        
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0
        
        return dot_product / (query_magnitude * doc_magnitude)
    
    def _get_query_vector(self, query: str) -> Dict[str, float]:
        """Convert query to TF-IDF vector."""
        words = self._preprocess_text(query)
        word_count = Counter(words)
        doc_count = len(self.documents)
        
        query_vector = {}
        for word in self.vocabulary:
            if word in word_count:
                tf = word_count[word] / len(words) if len(words) > 0 else 0
                # Use document frequency from existing index
                doc_freq = sum(1 for doc_vec in self.tfidf_vectors if doc_vec.get(word, 0) > 0)
                idf = math.log(doc_count / doc_freq) if doc_freq > 0 else 0
                query_vector[word] = tf * idf
            else:
                query_vector[word] = 0
        
        return query_vector
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for most relevant documents for a query."""
        if not self.documents or not self.tfidf_vectors:
            return []
        
        query_vector = self._get_query_vector(query)
        
        # Calculate similarity with all documents
        similarities = []
        for i, doc_vector in enumerate(self.tfidf_vectors):
            similarity = self._calculate_cosine_similarity(query_vector, doc_vector)
            similarities.append((self.documents[i], similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def answer_question(self, question: str) -> str:
        """Answer a question about cultural heritage."""
        if not self.documents:
            return "Sorry, I don't have access to my knowledge base. Please check if the data file exists."
        
        # Search for relevant documents
        results = self.search(question, top_k=2)
        
        if not results or results[0][1] < 0.1:  # Very low similarity threshold
            return ("I don't have specific information about that topic in my knowledge base. "
                   "You can ask me about famous cultural heritage sites like the Great Wall of China, "
                   "Machu Picchu, Egyptian Pyramids, Taj Mahal, Stonehenge, Petra, Colosseum, "
                   "Angkor Wat, Chichen Itza, or traditional Japanese architecture.")
        
        # Format the response
        best_match = results[0][0]
        response = f"**{best_match['title']}**\n\n{best_match['content']}"
        
        # Add additional context if second result is also relevant
        if len(results) > 1 and results[1][1] > 0.2:
            second_match = results[1][0]
            response += f"\n\n**Related: {second_match['title']}**\n{second_match['content'][:200]}..."
        
        return response
    
    def run_interactive(self):
        """Run the chatbot in interactive mode."""
        print("=" * 60)
        print("Cultural Heritage Chatbot")
        print("=" * 60)
        print("Ask me questions about cultural heritage sites and monuments!")
        print("Type 'quit', 'exit', or 'bye' to stop.")
        print("Type 'help' to see example questions.")
        print("-" * 60)
        
        while True:
            try:
                question = input("\nYou: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("Bot: Goodbye! Thanks for learning about cultural heritage!")
                    break
                
                if question.lower() == 'help':
                    print("\nBot: Here are some example questions you can ask:")
                    print("- Tell me about the Great Wall of China")
                    print("- What is Machu Picchu?")
                    print("- Who built the pyramids in Egypt?")
                    print("- What is special about Angkor Wat?")
                    print("- Tell me about Stonehenge")
                    print("- What is the Taj Mahal?")
                    print("- Describe the Colosseum")
                    print("- What is Japanese architecture like?")
                    continue
                
                answer = self.answer_question(question)
                print(f"\nBot: {answer}")
                
            except KeyboardInterrupt:
                print("\n\nBot: Goodbye!")
                break
            except Exception as e:
                print(f"\nBot: Sorry, I encountered an error: {e}")

def main():
    """Main function to run the chatbot."""
    chatbot = CulturalHeritageChatbot()
    
    if chatbot.documents:
        chatbot.run_interactive()
    else:
        print("Could not initialize chatbot. Please check your knowledge base file.")

if __name__ == "__main__":
    main()