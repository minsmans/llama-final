import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import json
from datetime import datetime

# Define the test prompt (Solar Flare example)
test_prompts = [
    {
        "instruction": (
            "event_title: Solar Flare Disrupts Global Communications\n"
            "event_description: A massive solar flare unleashed a geomagnetic storm, knocking out satellites and power grids worldwide.\n"
            "disaster_type: Solar Flare\n"
            "event_date_time: 2025-11-01 16:00:00 UTC\n"
            "event_location: Global\n"
            "event_coordinates: [0.0, 0.0, 0.0]\n"
            "disaster_details: class: X10, duration: 48 hours, alert_level: Red, affected_population: 8,000,000,000, damage_reports: 50 satellites lost, 10 million without power, GPS offline.\n"
            "climate_data: temperature: N/A, windspeed: N/A, winddirection: N/A, humidity: N/A, precipitation_probability: N/A, cloud_cover: N/A, pressure_sea_level: N/A"
        )
    }
]

# Load the fine-tuned model and tokenizer
model_path = "./finetuned_deepseek_lora"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16,
    max_new_tokens=2000,  # Allow for verbose output (1500–2000 tokens)
    temperature=0.7,  # Balanced creativity and coherence
    top_p=0.9,  # Nucleus sampling for better diversity
    do_sample=True  # Enable sampling for more natural output
)

# System prompt to reinforce the expected format and depth
system_prompt = (
    "You are a helpful assistant trained to provide detailed, structured disaster analysis reports. "
    "Output exactly these 7 sections without duplication or extra content, using structured text with angle-bracket headers "
    "(e.g., <disaster_analysis>, <event_summary>, etc.), with verbose, comprehensive details (1500–2000 tokens) matching the depth of the training examples: "
    "<disaster_analysis> (include 13 subsections like Key Data Points Extraction, Historical Patterns, etc.), "
    "<event_summary>, <detailed_analysis>, <predictions>, <impacts>, <mitigation_strategies>, <recommendations>. "
    "Do not output JSON; use structured text only."
)

# Function to extract disaster_type from the instruction
def extract_disaster_type(instruction):
    match = re.search(r"disaster_type:\s*([^\n]+)", instruction)
    return match.group(1).strip() if match else "Unknown"

# Function to validate the output format and content
def validate_output(output, disaster_type):
    validation_results = {"disaster_type": disaster_type, "format_errors": [], "content_warnings": []}

    # Check for JSON output
    if output.strip().startswith("[") or output.strip().startswith("{"):
        validation_results["format_errors"].append("Output is in JSON format; expected structured text with angle-bracket headers.")

    # Check for the 7 required sections
    required_sections = [
        "<disaster_analysis>", "</disaster_analysis>",
        "<event_summary>", "</event_summary>",
        "<detailed_analysis>", "</detailed_analysis>",
        "<predictions>", "</predictions>",
        "<impacts>", "</impacts>",
        "<mitigation_strategies>", "</mitigation_strategies>",
        "<recommendations>", "</recommendations>"
    ]
    for section in required_sections:
        if section not in output:
            validation_results["format_errors"].append(f"Missing section: {section}")

    # Check for the 13 subsections in <disaster_analysis>
    disaster_analysis_match = re.search(r"<disaster_analysis>(.*?)</disaster_analysis>", output, re.DOTALL)
    if disaster_analysis_match:
        disaster_analysis_content = disaster_analysis_match.group(1)
        subsections = re.findall(r"\d+\.\s*\*\*.*?\*\*:", disaster_analysis_content)
        if len(subsections) != 13:
            validation_results["format_errors"].append(
                f"<disaster_analysis> should have exactly 13 subsections; found {len(subsections)}."
            )
    else:
        validation_results["format_errors"].append("Could not parse <disaster_analysis> section.")

    # Content quality checks for Solar Flare
    if disaster_type == "Solar Flare":
        if "storm surge" in output.lower():
            validation_results["content_warnings"].append("Solar Flare output mentions 'storm surge', which is implausible (term is associated with floods/hurricanes).")
        if "killed" in output.lower() and not "secondary effects" in output.lower():
            validation_results["content_warnings"].append("Solar Flare output mentions direct deaths (e.g., 'killed'), which is implausible without secondary effects explained.")

    # Check for verbosity (rough token count)
    token_count = len(tokenizer.encode(output))
    if token_count < 1000:
        validation_results["content_warnings"].append(f"Output is too short ({token_count} tokens); expected 1500–2000 tokens for verbose, comprehensive details.")

    return validation_results

# Function to convert JSON output to structured text (fallback)
def convert_json_to_structured_text(output):
    try:
        json_output = json.loads(output.strip())
        if isinstance(json_output, list) and len(json_output) == 1:
            json_output = json_output[0]
        structured_text = ""
        for section in [
            "disaster_analysis", "event_summary", "detailed_analysis",
            "predictions", "impacts", "mitigation_strategies", "recommendations"
        ]:
            structured_text += f"<{section}>\n"
            structured_text += json_output.get(section, "Missing section content") + "\n"
            structured_text += f"</{section}>\n"
        return structured_text
    except json.JSONDecodeError:
        return output  # Return as-is if not valid JSON

# Generate and validate output for the test prompt
all_outputs = []
validation_summary = []
for test_case in test_prompts:
    instruction = test_case["instruction"]
    disaster_type = extract_disaster_type(instruction)

    # Construct the prompt
    prompt = f"<|user|>{system_prompt}\n{instruction}<|assistant>"

    # Generate the output
    print(f"Generating output for {disaster_type}...")
    generated = generator(
        prompt,
        max_new_tokens=2000,
        return_full_text=False,  # Only return the generated part (after <|assistant>)
        pad_token_id=tokenizer.pad_token_id
    )
    output = generated[0]["generated_text"]

    # Post-process: Convert JSON to structured text if necessary
    if output.strip().startswith("[") or output.strip().startswith("{"):
        output = convert_json_to_structured_text(output)

    # Validate the output
    validation_results = validate_output(output, disaster_type)
    validation_summary.append(validation_results)

    # Store the output with metadata
    all_outputs.append({
        "disaster_type": disaster_type,
        "prompt": prompt,
        "output": output,
        "validation_results": validation_results
    })

# Save the outputs to a file
output_file = f"test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for result in all_outputs:
        f.write(f"===== {result['disaster_type']} =====\n")
        f.write(f"Prompt:\n{result['prompt']}\n\n")
        f.write(f"Output:\n{result['output']}\n\n")
        f.write(f"Validation Results:\n")
        f.write(f"Format Errors: {result['validation_results']['format_errors']}\n")
        f.write(f"Content Warnings: {result['validation_results']['content_warnings']}\n")
        f.write("=" * 50 + "\n\n")

# Print a summary of validation results
print("\nValidation Summary:")
for result in validation_summary:
    print(f"\nDisaster Type: {result['disaster_type']}")
    print(f"Format Errors: {result['format_errors']}")
    print(f"Content Warnings: {result['content_warnings']}")
print(f"\nOutput saved to {output_file}")