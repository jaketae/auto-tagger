import argparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm


def word_counter(sentence):
    return len(sentence.split())


def chunkify(body, max_len, min_len):
    chunk = ""
    chunks = []
    word_count = 0
    sentences = body.split(".")
    for sentence in sentences:
        if not sentence:
            continue
        sentence += "."
        count = word_counter(sentence)
        if word_count <= max_len and word_count + count > max_len:
            chunks.append(chunk.lstrip())
            chunk = sentence
            word_count = count
        else:
            chunk += sentence
            word_count += count
    if chunk and word_count >= min_len:
        chunks.append(chunk.lstrip())
    return chunks


def get_urls():
    root = "https://jaketae.github.io/posts/"
    html = requests.get(root).text
    soup = BeautifulSoup(html, "html.parser")
    divs = soup.find_all("div", class_="list__item")
    return [f"https://jaketae.github.io{tag.find('a')['href']}" for tag in divs]


def get_all_tags(top_tags):
    root = "https://jaketae.github.io/tags/"
    html = requests.get(root).text
    soup = BeautifulSoup(html, "html.parser")
    ul = soup.find("ul", class_="taxonomy__index")
    return set(tuple(strong.text for strong in ul.find_all("strong"))[:top_tags])


class Parser:
    def __init__(self, url, all_tags, max_len, min_len):
        html = requests.get(url).text
        self.soup = BeautifulSoup(html, "html.parser")
        self.all_tags = all_tags
        self.max_len = max_len
        self.min_len = min_len

    def get_title(self):
        return self.soup.find("h1", {"id": "page-title"}).text.strip()

    def get_tags(self):
        tags = set()
        a_tags = self.soup.find_all("a", class_="page__taxonomy-item")
        for tag in a_tags:
            if "tags" in tag["href"]:
                tags.add(tag.text)
        return {tag: int(tag in tags) for tag in self.all_tags}

    def get_chunks(self):
        result = []
        p_tags = self.soup.find_all("p", class_="")
        for tag in p_tags:
            tmp = ""
            flag = True
            for code in tag.find_all("code"):
                code.extract()
            for char in tag.text:
                if char == "$":
                    flag = not flag
                    continue
                if flag:
                    tmp += char
            result.append(tmp)
        text = " ".join(result)
        chunks = chunkify(text, self.max_len, self.min_len)
        return chunks

    def parse(self):
        parsed = []
        title = self.get_title()
        chunks = self.get_chunks()
        tags = self.get_tags()
        if sum(tags.values()):
            for chunk in chunks:
                parsed.append({"title": title, "body": chunk, **tags})
            return parsed


def main(args):
    posts = []
    urls = get_urls()
    all_tags = get_all_tags(args.top_tags)
    for url in tqdm(urls):
        parser = Parser(url, all_tags, args.max_len, args.min_len)
        data = parser.parse()
        if data:
            posts.extend(data)
    df = pd.DataFrame(posts).set_index("title")
    df.to_csv(args.save_title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_len", type=int, default=256, help="maximum length of each text"
    )
    parser.add_argument(
        "--min_len", type=int, default=128, help="minimum length of each text"
    )
    parser.add_argument(
        "--top_tags", type=int, default=8, help="number of top tags to include",
    )
    parser.add_argument(
        "--save_title",
        type=str,
        default="data.csv",
        help="save title for generated csv",
    )
    args = parser.parse_args()
    main(args)
