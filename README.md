# AI-Powered Organization Simulation

Welcome to the AI-Powered Organization Simulation project. This project explores how an organization with AI-driven roles can communicate and work together.

## Table of Contents 📚

- [Getting Started](#getting-started)
- [Roles](#roles)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Simulation](#running-the-simulation)
- [Testing](#testing)
- [Contributing](#contributing)
- [Future Directions](#future-directions)

## Getting Started 🏁

The AI-Powered Organization Simulation is a Python-based project that simulates an organization with AI-driven roles, such as CEO, Ops, SE, and HR. The simulation demonstrates how these AI roles can communicate with each other and perform tasks specific to their roles.

## Roles 🧑‍💼

The organization has the following roles:

1. CEO (Human) - A human role responsible for making high-level decisions and setting the overall direction of the organization.
2. Ops (Operations) - A role responsible for executing escape codes when needed.
3. SE (Software Engineer) - A role responsible for designing, developing, and maintaining software applications, primarily creating escape codes when requested by other members of the organization.
4. HR (Human Resources) - A role responsible for managing AI resources and creating new roles within the organization.

## How It Works 🛠️

The simulation runs in a loop, where the organization members communicate with each other through messages. The System role can parse JSON blobs, store escape codes, and execute them when required. The code also contains a function to parse escape codes from a response text.

## Installation

To install the required packages for this project, run:

```bash
pip install -r requirements.txt
```

## Configuration

The runtime uses the OpenAI Python client and can target OpenAI or an OpenAI-compatible local endpoint.

Relevant environment variables:

- `OPENAI_API_KEY`: API key. Local compatible servers may accept any non-empty value.
- `OPENAI_BASE_URL` or `OPENAI_API_BASE`: optional compatible endpoint URL.
- `ROBITS_MODEL` or `OPENAI_MODEL`: default chat model.
- `ROBITS_CHEAP_MODEL` and `ROBITS_COSTLY_MODEL`: optional per-role overrides.

## Running the Simulation

To run the AI-Powered Organization Simulation:

1. Run the Python file using a Python interpreter:
   ```bash
   python main.py
   ```

For non-interactive smoke checks, provide an initial prompt and turn limit:

```bash
python main.py --prompt "Ops, say hello to HR" --turns 1
```

## Testing

The focused runtime tests do not call an external model service:

```bash
python -m unittest
```

## Contributing 🤝

We welcome contributions to this project! Feel free to submit pull requests, report bugs, or suggest new features. To get started, check out the [Future Directions](#future-directions) section for some ideas on how you can contribute.

## Future Directions 🚀

There are many exciting ways you can improve and expand this project. Here are a few ideas to get you started:

1. Add more roles to the organization (e.g., marketing, sales, or finance roles) to explore new interactions between AI-driven roles.
2. Enhance the AI's ability to understand context and engage in more complex conversations.
3. Implement a graphical user interface (GUI) for a more immersive and user-friendly simulation experience.
4. Explore ways to use real-world data to drive the simulation and make it more engaging and relevant.
5. Experiment with different AI models or techniques to improve the performance and capabilities of the AI roles.

Have fun exploring the AI-Powered Organization Simulation! We can't wait to see what you come up with! 🎉💡
