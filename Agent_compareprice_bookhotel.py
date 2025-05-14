#from typing import TypedDict, Optional
from typing_extensions import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from browser_use import Agent
import os
import asyncio

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key
)
class HotelState(TypedDict):
    agoda_result: Optional[str]
    trip_result: Optional[str]
    booking_result: Optional[str]
    best_hotel: Optional[dict]

# Node: Agoda
async def search_agoda(state):
    agent = Agent(
        task="""Go to https://www.agoda.com and Search hotels for:
        Phuket, Thailand | Check-in: 15 May | Check-out: 18 May | 1 Room | 3 Adults
        Return hotel names and prices in Thai Baht in json format of <Hotel Name> : <Price in Integer format eg. 1255>.""",
        llm=llm
    )
    state["agoda_result"] = await agent.run()
    print(state)
    return state

# Node: Trip.com
async def search_trip(state):
    agent = Agent(
        task="""Go to https://th.trip.com and Search hotels for:
        Phuket, Thailand | Check-in: 15 May | Check-out: 18 May | 1 Room | 3 Adults
        Return hotel names and prices in Thai Baht in json format of <Hotel Name> : <Price in Integer format eg. 1255>.""",
        llm=llm
    )
    state["trip_result"] = await agent.run()
    return state

# Node: Booking.com
async def search_booking(state):
    agent = Agent(
        task="""Go to https://www.booking.com/ and Search hotels for:
        Phuket, Thailand | Check-in: 15 May | Check-out: 18 May | 1 Room | 3 Adults
        Return hotel names and prices in Thai Baht.
        Output must only be in this format:
        {<Hotel Name> : <Price in Integer eg. 1255>}""",
        llm=llm
    )
    state["booking_result"] = await agent.run()
    return state

# Node: Compare
def compare_prices(state):
    def extract_prices(result, source):
        results = []
        final_result = result.final_result()
        lines = final_result.split(',')
        for line in lines:
            if ':' in line:
                hotel, price = line.split(':', 1)
                digits = ''.join(filter(str.isdigit, price))
                if digits:
                    results.append({
                        "source": source,
                        "hotel": hotel.strip(),
                        "price": int(digits)
                    })
        print(results)
        return results

    all_prices = []
    all_prices += extract_prices(state["agoda_result"], "Agoda")
    all_prices += extract_prices(state["trip_result"], "Trip.com")
    all_prices += extract_prices(state["booking_result"], "Booking.com")

    if all_prices:
        best = min(all_prices, key=lambda x: x["price"])
        state["best_hotel"] = best
    else:
        state["best_hotel"] = "No hotel found"
    return state

# (Optional) Node: Email
def send_email(state):
    best = state["best_hotel"]
    print(f"\n📧 Email sent: {best}")
    return state


graph = StateGraph(HotelState)
graph.add_node("search_agoda", search_agoda)
graph.add_node("search_trip", search_trip)
graph.add_node("search_booking", search_booking)
graph.add_node("compare_prices", compare_prices)
graph.add_node("send_email", send_email)

graph.set_entry_point("search_agoda")
graph.add_edge("search_agoda", "search_trip")
graph.add_edge("search_trip", "search_booking")
graph.add_edge("search_booking", "compare_prices")
graph.add_edge("compare_prices", "send_email")
graph.add_edge("send_email", END)

compiled_graph = graph.compile()

# Run it
async def run():
    final_state = await compiled_graph.ainvoke({})
    print("\n✅ Final Output:", final_state["best_hotel"])

if __name__ == "__main__":
    asyncio.run(run())
