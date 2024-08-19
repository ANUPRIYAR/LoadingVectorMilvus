from langchain.document_loaders import SitemapLoader, WebBaseLoader
import nest_asyncio

nest_asyncio.apply()

output_file = open("scraped_content.txt", "w", encoding="utf-8")
# sitemap_loader = SitemapLoader(web_path="https://langchain.readthedocs.io/sitemap.xml",
#             filter_urls = ["https://api.python.langchain.com/en/latest"],
#             )
# sitemap_loader = SitemapLoader(web_path="https://www.pipiads.com/sitemap_examples_100.xml")

loader= WebBaseLoader("https://python.langchain.com/docs/modules/agents/")

docs = loader.load()
loader.requests_per_second = 2
loader.requests_kwargs = {"verify": False}

print(docs)
for doc in docs:
    output_file.write(str(doc))

output_file.close()