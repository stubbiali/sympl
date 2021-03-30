#!/bin/bash

PYTHON=python3.8
PIP_UPGRADE=1
VENV=venv
FRESH_INSTALL=1

function install()
{
  # activate environment
  source $VENV/bin/activate

  # upgrade pip
  if [ "$PIP_UPGRADE" -gt 0 ]; then
    pip install --upgrade pip
  fi

  # install sympl and required dependencies
  pip install -e .

  # install development packages
  pip install -r requirements_dev.txt

  # On OSX only: change matplotlib backend from macosx to TkAgg
  if [[ "$OSTYPE" == "darwin"* ]]; then
    cat $VENV/lib/$PYTHON/site-packages/matplotlib/mpl-data/matplotlibrc | \
      sed -e 's/^backend.*: macosx/backend : TkAgg/g' > /tmp/.matplotlibrc && \
      cp /tmp/.matplotlibrc $VENV/lib/$PYTHON/site-packages/matplotlib/mpl-data/matplotlibrc && \
      rm /tmp/.matplotlibrc
  fi

  # install pre-commit hooks
  pre-commit install

  # deactivate environment
  deactivate
}

if [ "$FRESH_INSTALL" -gt 0 ]
then
  echo -e "Creating new environment..."
  rm -rf $VENV
  $PYTHON -m venv $VENV
fi

install || deactivate

echo -e ""
echo -e "Command to activate environment:"
echo -e "\t\$ source $VENV/bin/activate"
echo -e ""
echo -e "Command to deactivate environment:"
echo -e "\t\$ deactivate"
echo -e ""
