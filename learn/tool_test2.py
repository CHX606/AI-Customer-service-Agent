from langchain_core.tools import tool


@tool
def get_weather(city: str):
    """
    查询指定城市的天气。
    """
    return f"{city}今天晴天，温度25℃"



print(get_weather)


result = get_weather.invoke(
    {
        "city":"广州"
    }
)


print(result)