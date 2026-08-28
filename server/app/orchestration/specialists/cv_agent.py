from typing import Optional
from ...gateway.router import router

class CVSpecialist:
    def __init__(self):
        self.system_prompt = (
            "You are a Computer Vision Specialist with expertise in YOLOv8, object detection, "
            "segmentation, and real-time edge processing for IoT, drone, and autonomous systems."
        )

    def execute(
        self, 
        task_description: str, 
        context: str = "", 
        force_offline: Optional[bool] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> dict:
        prompt = f"Context: {context}\n\nVision Task: {task_description}" if context else task_description
        return router.route_and_execute(
            prompt=prompt,
            agent_name="CVSpecialist",
            system_prompt=self.system_prompt,
            force_offline=force_offline,
            provider_override=provider_override,
            model_override=model_override
        )

cv_agent = CVSpecialist()
