# Artist Portolio Website Crawler and Search Engine

---

This is a webcrawler + data processer to index artist portfolio website information to allow search based on:
- color profile (up to 5, order matches with percentage descendingly)
- color profile match threshold (how accurately the website's color profile need to match the query)
- rating (based on my personal taste of the website's screenshot, 0-100, I rated 2000 and used a CNN model to rate the rest)
- text density (text pixels to non text pixels ratio)
- color variance
- layout complexity (base don edge detection)
- category (empty, miminal text, complex design, highliy detailed) (classified based on the above 3 features)
- keyword (matches for the whole word in the url, only one keyowrd allowed)

Each entry displays:
- website screenshot (taken by selenium with mozila geckodriver)
- rating (rated by CNN model based on screenshot)
- color profile (top 5 most used colors and percentage in the website's screenshot)
- website url (may not be the exact url, might need to search for it on a search engine)

---

I tried different strategies to crawl the urls:
- select top 100 search engine results of "artist portfolio websites" and crawl for portfolio website urls in those webpages, crawl depth = 1, but most are repeated and are very limited in numbers
- crawl specific artist portfolio showcase websites, such as webflow, but they have very limited amount
- the most efficient method, use database of contemporary artist names and crawl for their websites by filtering top 5 results of search engine searching for their name
All of these methods are lmited in their ability to filter out random, unrelated websites, thus my database has lots of irrelevant websites

---

I was inspired to make this project because I was helping my friend make an artist portfolio website, so I wanted to gather some inspirations. I hope this project could be helpful to you in the same way as well.

---
The database currently has 13000 entires of artist portolio websites, some of which happen to be irrelevant, you can help me remove them by pressing the "delete" button.
The website is hosted live at https://portfolios.hopto.org/
The portfolio entries are in JSON format availablea at https://portfolios.hopto.org/api/portfolios
---

This project is not being actively maintained. The crawler code needs to be adjusted to be usable.
