**Assignment 2 — Create an Agent (Top-3 Web Sources)**

**What this agent does**
- Takes a user question
- (Optionally) refines it into a better search query
- Uses `internet_search()` to get ranked results
- Fetches content from the **top 3 URLs** using `fetch_url()`
- Synthesizes an answer grounded **only** in those 3 pages
- Provides citations as markdown links: `[Title](URL)`

---

**Setup**

1) Create an OpenRouter account: https://openrouter.ai

2) Set your API key:

Mac/Linux:
```bash
export OPENROUTER_API_KEY="YOUR_KEY_HERE"
