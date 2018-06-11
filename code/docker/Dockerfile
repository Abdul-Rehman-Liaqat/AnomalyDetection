FROM jupyter/scipy-notebook

LABEL maintainer="Abdul Rehman Liaqat <liaqat@uni-hildesheim.de>"

RUN conda install --quiet --yes \
    'tensorflow=1.5*' \
    'keras=2.1*' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
