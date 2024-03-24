import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    corpus_size = len(corpus)
    num_links = len(corpus[page])
    distribution = {}

    if num_links == 0:
        distribution = {page: 1 / corpus_size for page in corpus}
        return distribution
 
    for p in corpus:
        value = (1 - damping_factor) / corpus_size
        if p not in corpus[page]:
            distribution[p] = value
        else:
            value += damping_factor / num_links
            distribution[p] = value

    return distribution

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageranks = {page: 0 for page in corpus}
    page = random.choice(list(corpus.keys()))
    
    for i in range(n):
        distribution = transition_model(corpus, page, damping_factor)
        page = random.choices(
            list(distribution.keys()),
            list(distribution.values()),
            k=1
        )[0]
        pageranks[page] += 1
        
    for page in pageranks:
        pageranks[page] /= n

    return pageranks

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    convergence_threshold = 0.001
    corpus_size = len(corpus)
    pageranks = {page: 1 / corpus_size for page in corpus}
    
    while True:
        new_pageranks = {}
        for page in corpus:
            new_pagerank = (1 - damping_factor) / corpus_size
            source_pages = [p for p in corpus if page in corpus[p] or len(corpus[p]) == 0]
            new_pagerank += damping_factor * sum((pageranks[p] / len(corpus[p])) if len(corpus[p]) != 0 else (pageranks[p] / len(corpus)) for p in source_pages)
            new_pageranks[page] = new_pagerank

        max_change = max(abs(new_pageranks[page] - pageranks[page]) for page in corpus)
        if max_change < convergence_threshold:
            break

        pageranks = new_pageranks

    return pageranks


if __name__ == "__main__":
    main()
