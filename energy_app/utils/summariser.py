from ..utils.chat import chat

def summarise(name: str, preds, stats: dict):
    prompt = (
        f"You are an energy-data analyst.\n"
        f"Dataset: {name}\n"
        f"Key stats: {stats}\n"
        f"Next-period predictions: {preds.tolist()[:10]}...\n"
        f"Write 2 concise paragraphs explaining expected trends, uncertainty, "
        f"and any actionable insight."
    )
    return chat([{"role": "user", "content": prompt}])
