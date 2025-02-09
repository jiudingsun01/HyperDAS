ARG PYTHON_VERSION
ARG GIT_NAME
ARG GIT_EMAIL
ARG PROJECT_NAME

FROM ghcr.io/${GIT_NAME}/ml-base:py${PYTHON_VERSION}

ENV PATH="/usr/local/bin:/root/.local/bin:/opt/conda/bin:${PATH}"
WORKDIR /workspace/${PROJECT_NAME}

# Setup Git configuration
RUN git config --global user.email "${GIT_EMAIL}" && \
    git config --global user.name "${GIT_NAME}"

# Install requirements
COPY requirements.txt /tmp/
RUN uv pip install --system --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt || true

# If python package exists, install it
RUN python setup.py develop || true \
    && uv pip install --system --no-cache-dir -e . || true

COPY scripts/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["zsh"]