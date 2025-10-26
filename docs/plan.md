# Execution plan

## Task

You work for a bank, which experienced a systems failure.
The loan approval system is unavailable and you're tasked
with developing an algorithm that approves or rejects loans instead.
The bank uses an old legacy IT system, which means that
you won't be able to deploy the algorithm within the IT system. Instead,
you need to develop the algorithm outside the existing IT system
and then integrate it.

## Thinking

The dataset contains loan approval data.

- I want to set up an API that receives the applicant data as the payload
  and sends either "approval" or "rejection" as a response (boolean classification).
- Since the algorithm needs to be available as quickly as possible, I'll take
  as many reasonable shortcuts as I can.

### Repo setup

- only allow squash & merge on Github
- add .pre-commit hooks

### EDA

- [Install pixi](https://pixi.sh/latest/installation/)
- `pixi add python`, `pixi add jupyter lab`

### Modelling

- Run python scripts within the pixi shell. Run `pixi shell`.

### REST API

- Start the server with `uvicorn src.app:app --reload`
- Add Dockerfile
- Add dev environment to avoid having jupyter lab in the Docker deployment
