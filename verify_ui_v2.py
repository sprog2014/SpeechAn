import asyncio
from playwright.async_api import async_playwright
import os
import time

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Переходим на страницу
        try:
            await page.goto("http://localhost:8501")

            # Логин
            await page.fill("input[type='password']", "admin")
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(2000)

            # Переход в настройки
            # В Streamlit боковой панели ссылки - это кнопки или ссылки с текстом
            await page.click("text=settings")
            await page.wait_for_timeout(2000)

            # Скриншот
            os.makedirs("docs/images", exist_ok=True)
            await page.screenshot(path="docs/images/settings_interface.png", full_page=True)
            print("Screenshot saved to docs/images/settings_interface.png")

        except Exception as e:
            print(f"Error: {e}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
