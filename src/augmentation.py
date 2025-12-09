import random
import nltk
from nltk.corpus import wordnet, stopwords

# Ensure resources are downloaded
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

def get_synonyms(word):
    syns = set()
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            w = lem.name().replace('_', ' ').lower()
            if w != word:
                syns.add(w)
    return list(syns)

def synonym_replacement(words, n):
    if not words: return words
    new_words = words.copy()
    candidates = [w for w in words if w not in STOP_WORDS]
    random.shuffle(candidates)
    num_replaced = 0
    for w in candidates:
        syns = get_synonyms(w)
        if syns:
            new_words = [random.choice(syns) if x == w else x for x in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

def random_insertion(words, n):
    if not words: return words
    new_words = words.copy()
    for _ in range(n):
        candidates = [w for w in new_words if w not in STOP_WORDS]
        if not candidates: break
        w = random.choice(candidates)
        syns = get_synonyms(w)
        if not syns: continue
        new_words.insert(random.randint(0, len(new_words)), random.choice(syns))
    return new_words

def random_swap(words, n):
    if len(words) < 2: return words
    new_words = words.copy()
    for _ in range(n):
        i, j = random.sample(range(len(new_words)), 2)
        new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words

def random_deletion(words, p):
    if len(words) <= 1: return words
    new_words = [w for w in words if random.random() > p]
    return new_words if new_words else [random.choice(words)]

def augment_sentence(sentence, alpha=0.1, n_aug=4):
    """
    Applies EDA (Easy Data Augmentation) strategies to a sentence.
    """
    words = sentence.split()
    l = len(words)
    if l == 0: return []
    n = max(1, int(alpha * l))
    
    ops = [
        lambda w: synonym_replacement(w, n),
        lambda w: random_insertion(w, n),
        lambda w: random_swap(w, n),
        lambda w: random_deletion(w, alpha)
    ]
    
    augmented = []
    for _ in range(n_aug):
        op = random.choice(ops)
        aug_words = op(words)
        augmented.append(" ".join(aug_words))
    return augmented