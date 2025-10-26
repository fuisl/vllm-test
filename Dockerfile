FROM vllm/vllm-openai:v0.11.0

RUN groupadd -g 1006 fuisloy && useradd -m -u 1006 -g 1006 fuisloy
RUN usermod -aG sudo fuisloy

RUN mkdir -p /home/fuisloy/.cache && chmod -R 777 /home/fuisloy/.cache

RUN id fuisloy