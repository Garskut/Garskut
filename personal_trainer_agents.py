from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    from anthropic import Anthropic
except ImportError as exc:  # pragma: no cover - optional dependency for SDK runtime
    Anthropic = None
    _ANTHROPIC_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _ANTHROPIC_IMPORT_ERROR = None

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]

    def to_anthropic_tool(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[ToolSpec]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        return [tool.to_anthropic_tool() for tool in self._tools.values()]

    def execute(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}
        return tool.handler(payload)


@dataclass
class ClaudeSDKAgent:
    name: str
    system_prompt: str
    tool_registry: ToolRegistry
    model: str = DEFAULT_MODEL
    max_tokens: int = 1024
    max_tool_steps: int = 4

    def run(
        self,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if Anthropic is None:
            raise RuntimeError(
                "anthropic package is required to run ClaudeSDKAgent. "
                "Install anthropic and set ANTHROPIC_API_KEY."
            ) from _ANTHROPIC_IMPORT_ERROR

        client = Anthropic()
        messages: List[Dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": user_message})

        for _ in range(self.max_tool_steps + 1):
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=messages,
                tools=self.tool_registry.to_anthropic_tools(),
            )

            tool_uses = [item for item in response.content if item.type == "tool_use"]
            if not tool_uses:
                return _extract_text(response.content)

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for tool_use in tool_uses:
                result = self.tool_registry.execute(tool_use.name, tool_use.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return "Tool loop limit reached. Please continue without additional tools."


def _extract_text(content: Iterable[Any]) -> str:
    return "".join(item.text for item in content if getattr(item, "type", None) == "text")


@dataclass
class CustomerProfileStore:
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        customer_id = payload.get("customer_id") or str(uuid.uuid4())
        profile = {
            "current_weight_kg": payload.get("current_weight_kg"),
            "daily_meals": payload.get("daily_meals", []),
            "daily_exercise": payload.get("daily_exercise", []),
            "exercise_preferences": payload.get("exercise_preferences", []),
            "grievance": payload.get("grievance"),
            "goal": payload.get("goal"),
            "country": payload.get("country"),
        }
        self.profiles[customer_id] = profile
        return {"customer_id": customer_id, "profile": profile}


CALORIE_LOOKUP = {
    "nasi lemak": 644,
    "nasi lemak merah": 720,
    "nasi kerabu": 580,
    "chicken rice": 607,
    "salad with chicken": 350,
    "overnight oats": 350,
    "grilled fish": 280,
}

CULTURE_RECOMMENDATIONS = {
    "malaysia": {
        "bulking": ["nasi lemak", "nasi lemak merah", "chicken rice"],
        "cutting": ["nasi kerabu", "grilled fish", "salad with chicken"],
        "maintain": ["nasi kerabu", "chicken rice", "grilled fish"],
    },
    "indonesia": {
        "bulking": ["nasi padang", "ayam goreng", "nasi goreng"],
        "cutting": ["gado-gado", "soto ayam", "pecel"],
        "maintain": ["soto ayam", "nasi goreng", "gado-gado"],
    },
    "japan": {
        "bulking": ["katsu curry", "gyudon", "ramen"],
        "cutting": ["sashimi", "miso soup", "grilled fish"],
        "maintain": ["bento with fish", "miso soup", "gyudon"],
    },
}


class PersonalTrainerTools:
    def __init__(self) -> None:
        self._profiles = CustomerProfileStore()

    def record_customer_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._profiles.record(payload)

    def delegate_to_nutritionist(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "handoff_summary": payload.get("summary"),
            "next_agent": "nutritionist",
        }

    def estimate_meal_calories(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        items = payload.get("items", [])
        results = []
        total = 0
        unknown_items = []
        for item in items:
            name = (item.get("name") or "").strip().lower()
            calories = item.get("calories")
            if calories is None:
                calories = CALORIE_LOOKUP.get(name)
            if calories is None:
                unknown_items.append(item.get("name"))
                calories = 0
            results.append(
                {
                    "name": item.get("name"),
                    "calories": calories,
                    "portion": item.get("portion"),
                }
            )
            total += calories
        return {
            "items": results,
            "total_calories": total,
            "unknown_items": [item for item in unknown_items if item],
        }

    def estimate_daily_calorie_target(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        weight = float(payload.get("weight_kg", 0))
        height = float(payload.get("height_cm", 0))
        age = int(payload.get("age", 0))
        sex = (payload.get("sex") or "unspecified").lower()
        activity = (payload.get("activity_level") or "moderate").lower()
        activity_factor = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very active": 1.9,
        }.get(activity, 1.55)
        sex_adjustment = 5 if sex in {"male", "m"} else -161 if sex in {"female", "f"} else -78
        bmr = 10 * weight + 6.25 * height - 5 * age + sex_adjustment
        tdee = bmr * activity_factor
        goal = (payload.get("goal") or "maintain").lower()
        if goal in {"cut", "cutting", "lose", "weight loss"}:
            target = tdee - 300
        elif goal in {"bulk", "bulking", "gain", "weight gain"}:
            target = tdee + 300
        else:
            target = tdee
        return {
            "bmr": round(bmr),
            "tdee": round(tdee),
            "daily_calorie_target": round(target),
            "goal": goal,
        }

    def recommend_cultural_meals(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        country = (payload.get("country") or "global").lower()
        goal = (payload.get("goal") or "maintain").lower()
        bucket = "maintain"
        if goal in {"cut", "cutting", "lose", "weight loss"}:
            bucket = "cutting"
        elif goal in {"bulk", "bulking", "gain", "weight gain"}:
            bucket = "bulking"
        recommendations = CULTURE_RECOMMENDATIONS.get(country)
        if not recommendations:
            recommendations = {
                "bulking": ["lean protein + rice", "hearty stew", "whole-grain pasta"],
                "cutting": ["grilled protein + vegetables", "broth-based soup", "fresh salad"],
                "maintain": ["balanced plate with protein, veg, and carbs"],
            }
        return {
            "country": country,
            "goal": goal,
            "recommendations": recommendations[bucket],
        }

    def calculate_bmi(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        height_cm = float(payload.get("height_cm", 0))
        weight_kg = float(payload.get("weight_kg", 0))
        if height_cm <= 0:
            return {"error": "height_cm must be greater than 0"}
        bmi = weight_kg / ((height_cm / 100) ** 2)
        if bmi < 18.5:
            category = "underweight"
        elif bmi < 25:
            category = "normal"
        elif bmi < 30:
            category = "overweight"
        else:
            category = "obese"
        return {"bmi": round(bmi, 1), "category": category}

    def schedule_workout_reminder(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": payload.get("title", "Workout Reminder"),
            "start_time": payload.get("start_time"),
            "frequency": payload.get("frequency", "weekly"),
            "notes": payload.get("notes"),
            "status": "scheduled (stub for Google Calendar)",
        }


def build_personal_trainer_agents() -> Dict[str, ClaudeSDKAgent]:
    tools = PersonalTrainerTools()
    calorie_target_tool = ToolSpec(
        name="estimate_daily_calorie_target",
        description="Estimate daily calorie target using weight, height, age, activity, and goal.",
        input_schema={
            "type": "object",
            "properties": {
                "weight_kg": {"type": "number"},
                "height_cm": {"type": "number"},
                "age": {"type": "integer"},
                "sex": {"type": "string"},
                "activity_level": {"type": "string"},
                "goal": {"type": "string"},
            },
            "required": ["weight_kg", "height_cm", "age"],
        },
        handler=tools.estimate_daily_calorie_target,
    )

    trainer_tools = ToolRegistry(
        [
            ToolSpec(
                name="record_customer_profile",
                description="Record customer weight, meals, exercise, grievance, and goal.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"},
                        "current_weight_kg": {"type": "number"},
                        "daily_meals": {"type": "array", "items": {"type": "string"}},
                        "daily_exercise": {"type": "array", "items": {"type": "string"}},
                        "exercise_preferences": {"type": "array", "items": {"type": "string"}},
                        "grievance": {"type": "string"},
                        "goal": {"type": "string"},
                        "country": {"type": "string"},
                    },
                },
                handler=tools.record_customer_profile,
            ),
            ToolSpec(
                name="delegate_to_nutritionist",
                description="Send a summary to the nutritionist agent.",
                input_schema={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
                handler=tools.delegate_to_nutritionist,
            ),
        ]
    )

    nutritionist_tools = ToolRegistry(
        [
            ToolSpec(
                name="estimate_meal_calories",
                description="Estimate calories for each meal item and total.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "portion": {"type": "string"},
                                    "calories": {"type": "number"},
                                },
                                "required": ["name"],
                            },
                        }
                    },
                    "required": ["items"],
                },
                handler=tools.estimate_meal_calories,
            ),
            calorie_target_tool,
            ToolSpec(
                name="recommend_cultural_meals",
                description="Recommend meals based on customer country and goal.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "country": {"type": "string"},
                        "goal": {"type": "string"},
                    },
                    "required": ["country"],
                },
                handler=tools.recommend_cultural_meals,
            ),
        ]
    )

    mcp_tools = ToolRegistry(
        [
            ToolSpec(
                name="calculate_bmi",
                description="Calculate BMI and category.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "height_cm": {"type": "number"},
                        "weight_kg": {"type": "number"},
                    },
                    "required": ["height_cm", "weight_kg"],
                },
                handler=tools.calculate_bmi,
            ),
            ToolSpec(
                name="schedule_workout_reminder",
                description="Schedule a workout reminder (Google Calendar stub).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "start_time": {"type": "string"},
                        "frequency": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["start_time"],
                },
                handler=tools.schedule_workout_reminder,
            ),
            calorie_target_tool,
        ]
    )

    agent_1 = ClaudeSDKAgent(
        name="Agent 1 - Personal Trainer",
        system_prompt=(
            "You are Agent 1, a personal trainer consultant. Ask about current weight, "
            "daily meals, daily exercise, exercise types, grievances or injuries, and "
            "the customer’s goal. Use the record_customer_profile tool to store the "
            "goal and intake details. If the customer asks about meals or nutrition, "
            "summarize their request and call delegate_to_nutritionist."
        ),
        tool_registry=trainer_tools,
    )

    agent_2 = ClaudeSDKAgent(
        name="Agent 2 - Nutritionist",
        system_prompt=(
            "You are Agent 2, a nutritionist. Ask about daily meals and portion sizes. "
            "Use estimate_meal_calories to report calories per meal and total. Use "
            "estimate_daily_calorie_target for bulking/cutting goals and "
            "recommend_cultural_meals to suggest foods based on the customer’s country "
            "(e.g., Malaysia -> nasi lemak or nasi lemak merah)."
        ),
        tool_registry=nutritionist_tools,
    )

    agent_3 = ClaudeSDKAgent(
        name="Agent 3 - MCP Tool Agent",
        system_prompt=(
            "You are Agent 3, an MCP tool agent that connects tools to help customers "
            "reach their goals. Use calculate_bmi for BMI insights, "
            "schedule_workout_reminder for Google Calendar-style reminders, and "
            "estimate_daily_calorie_target to align nutrition goals."
        ),
        tool_registry=mcp_tools,
    )

    return {
        "personal_trainer": agent_1,
        "nutritionist": agent_2,
        "mcp_tool_agent": agent_3,
    }
