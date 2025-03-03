ARG PYTHON_VERSION
ARG GIT_NAME
ARG GIT_EMAIL
ARG PROJECT_NAME
ARG VARIANT=cuda

FROM ghcr.io/${GIT_NAME}/ml-base:py${PYTHON_VERSION}-${VARIANT}

ENV PATH="/usr/local/bin:/root/.local/bin:/opt/conda/bin:${PATH}"
WORKDIR /workspace/${PROJECT_NAME}

# Setup Git configuration
RUN git config --global user.email "${GIT_EMAIL}" && \
    git config --global user.name "${GIT_NAME}"

# Install requirements
COPY requirements.txt /tmp/
RUN uv pip install --system --no-cache-dir --no-upgrade -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt || true

# If python package exists, install it
RUN python setup.py develop || true \
    && uv pip install --system --no-cache-dir --no-upgrade -e . || true \
    && uv pip install --system --no-cache-dir --no-upgrade ipdb

COPY scripts/entrypoint.sh scripts/install.sh* /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Run install script if exists
RUN if [ -f /usr/local/bin/install.sh ]; then bash /usr/local/bin/install.sh; fi

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["zsh"]
