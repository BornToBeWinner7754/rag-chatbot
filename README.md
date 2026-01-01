Docker Concepts
Dockerfile > Recipe to build your app
Image > Built app (immutable)
Container > Running instance of image
Docker engine > Software that runs container
Port mapping > Connect container to your PC

Image = Blueprint

| Line                    | Meaning                          |
| ----------------------- | -------------------------------- |
| `FROM python:3.11-slim` | Start with lightweight Python    |
| `WORKDIR /app`          | Inside container, work in `/app` |
| `COPY . /app`           | Copy your project into container |
| `RUN pip install ...`   | Install dependencies             |
| `EXPOSE 80`             | App listens on port 80           |
| `CMD ...`               | Start FastAPI app                |
