# standard library imports
import os
import time

# third party imports
import arxiv
import pandas as pd
from tqdm import tqdm

# local imports


class ArXivScraper:
    def __init__(self) -> None:
        self.categories = [
            "stat.AP",
            "stat.CO",
            "stat.ML",
            "stat.ME",
            "stat.TH",
        ]
        self.save_path = os.path.join(os.path.dirname(__file__), "data", "raw_data.csv")

    def get_papers_by_category(self, category: str) -> pd.DataFrame:
        client = arxiv.Client(
            page_size=1000,
            delay_seconds=5,
            num_retries=10,
        )

        query = f"cat:{category}"
        search = arxiv.Search(
            query=query,
            max_results=50000,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        results = client.results(search)

        titles = []
        abstracts = []
        publish_dates = []
        primary_categories = []

        for result in tqdm(results, desc=f"Downloading {category} papers"):
            primary_category = result.primary_category
            if primary_category == "math.ST":
                primary_category = "stat.TH"

            if primary_category == category:
                titles.append(result.title)
                abstracts.append(result.summary)
                publish_dates.append(result.published)
                primary_categories.append(primary_category)

        df = pd.DataFrame(
            {
                "Title": titles,
                "Abstract": abstracts,
                "Publish Date": publish_dates,
                "Primary Category": primary_categories,
            }
        )

        return df

    def __call__(self) -> None:
        dfs = []
        for category in self.categories:
            df = self.get_papers_by_category(category)
            dfs.append(df)
            time.sleep(30)

        raw_data = pd.concat(dfs).sort_values(by=["Publish Date"])
        raw_data.to_csv(self.save_path, index=False)
