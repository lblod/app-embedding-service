import re
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

class TextChunker:
    """
    A class used to chunk text in various ways.
    """

    @staticmethod
    def chunk_by_max_words(text, max_words):
        """
        Splits the text into chunks of at most max_words words.
        """
        words = text.split()
        return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    @staticmethod
    def chunk_by_html_tags(text, html_tag='p'):
        """
        Splits the text into chunks based on HTML tags.
        """
        try:
            soup = BeautifulSoup(text, 'html.parser')
            return [p.get_text(strip=True) for p in soup.find_all(html_tag)]
        except Exception as e:
            print(f"Failed to parse HTML: {e}")
            return []

    @staticmethod
    def chunk_by_symbols(text, symbols):
        """
        Splits the text into chunks based on a list of symbols.
        """
        if not symbols:
            return [text]
        pattern = '|'.join(re.escape(symbol) for symbol in symbols)
        chunks = re.split(pattern, text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    @staticmethod
    def chunk_by_sentences(text):
        """
        Splits the text into chunks where each chunk is a sentence.
        """
        return sent_tokenize(text)

    @staticmethod
    def chunk_by_paragraphs(text):
        """
        Splits the text into chunks where each chunk is a paragraph.
        """
        return text.split('\n')
    