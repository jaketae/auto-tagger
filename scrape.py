import argparse
import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


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
    def __init__(self, url, all_tags):
        html = requests.get(url).text
        self.soup = BeautifulSoup(html, "html.parser")
        self.all_tags = all_tags

    def get_title(self):
        return self.soup.find("h1", {"id": "page-title"}).text.strip()

    def get_tags(self):
        tags = set()
        a_tags = self.soup.find_all("a", class_="page__taxonomy-item")
        for tag in a_tags:
            if "tags" in tag["href"]:
                tags.add(tag.text)
        return {tag: int(tag in tags) for tag in self.all_tags}

    def get_body(self):
        result = []
        p_tags = self.soup.find_all("p", class_="")
        for p in p_tags:
            for remove_type in ("code", "script", "span"):
                for remove_tag in p.find_all(remove_type):
                    remove_tag.decompose()
            result.append(p.text.strip())
        text = " ".join(result)
        for regexp in (r"\$.*?\$", r"\\\(.*?\\\)", r"\[.*?\]"):
            body = re.sub(regexp, "", text)
        return body

    def parse(self):
        title = self.get_title()
        body = self.get_body()
        tags = self.get_tags()
        if sum(tags.values()):
            return {"title": title, "body": body, **tags}


def main(args):
    posts = []
    urls = get_urls()
    all_tags = get_all_tags(args.top_tags)
    for url in tqdm(urls):
        parser = Parser(url, all_tags)
        data = parser.parse()
        if data is not None:
            posts.append(data)
    total_df = pd.DataFrame(posts).set_index("title")
    tv_df, test_df = train_test_split(total_df, test_size=args.test_size)
    train_df, val_df = train_test_split(tv_df, test_size=args.val_size)
    target_dir = "data"
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    for title, df in zip(("train", "val", "test"), (train_df, val_df, test_df)):
        df.to_csv(os.path.join(target_dir, f"{title}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top_tags", type=int, default=8, help="number of top tags to include",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.15, help="size of test samples"
    ),
    parser.add_argument(
        "--val_size", type=float, default=0.15, help="size of validation samples"
    ),
    args = parser.parse_args()
    main(args)
