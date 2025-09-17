import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import json

URL = " "

async def scrape_table_from_gofin():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto(URL)
        await page.wait_for_selector("table.spis")

        content = await page.content()
        await browser.close()

        soup = BeautifulSoup(content, "html.parser")
        table = soup.find("table", class_="spis")
        rows = table.find_all("tr")[1:]  # Pomijamy nagłówek

        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            kod = cols[0].get_text(strip=True)
            nazwa = cols[1].get_text(strip=True)
            opis_elem = cols[2] if len(cols) > 2 else None
            opis = ""

            if opis_elem:
                # Zbiera tekst i listy <ul> jako ciągły opis
                opis = " ".join(opis_elem.stripped_strings)

            data.append({
                "kod": kod,
                "nazwa": nazwa,
                "opis": opis
            })

        with open("pkd_dzial_13_tabela.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Zapisano {len(data)} rekordów do 'pkd_dzial_13_tabela.json'")

asyncio.run(scrape_table_from_gofin())
