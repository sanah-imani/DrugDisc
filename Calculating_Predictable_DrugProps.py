{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Calculating Predictable DrugProps.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sanah-imani/DrugDisc/blob/master/Calculating_Predictable_DrugProps.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1bl3mViuxpIQ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from matplotlib import pyplot as plt\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "%matplotlib inline"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "sKcAxOiS8MBK",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!pip install chembl_webresource_client"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cX0HDMla8J-R",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import gevent.monkey"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6oFg7BVr86V4",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "gevent.monkey.patch_all()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "N0RJw4yD8w-4",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from chembl_webresource_client.new_client import new_client"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-1HlzCIfpaWW",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import sys\n",
        "import os\n",
        "import requests\n",
        "import subprocess\n",
        "import shutil\n",
        "from logging import getLogger, StreamHandler, INFO\n",
        "\n",
        "\n",
        "logger = getLogger(__name__)\n",
        "logger.addHandler(StreamHandler())\n",
        "logger.setLevel(INFO)\n",
        "\n",
        "\n",
        "def install(\n",
        "        chunk_size=4096,\n",
        "        file_name=\"Miniconda3-latest-Linux-x86_64.sh\",\n",
        "        url_base=\"https://repo.continuum.io/miniconda/\",\n",
        "        conda_path=os.path.expanduser(os.path.join(\"~\", \"miniconda\")),\n",
        "        rdkit_version=None,\n",
        "        add_python_path=True,\n",
        "        force=False):\n",
        "    \"\"\"install rdkit from miniconda\n",
        "    ```\n",
        "    import rdkit_installer\n",
        "    rdkit_installer.install()\n",
        "    ```\n",
        "    \"\"\"\n",
        "\n",
        "    python_path = os.path.join(\n",
        "        conda_path,\n",
        "        \"lib\",\n",
        "        \"python{0}.{1}\".format(*sys.version_info),\n",
        "        \"site-packages\",\n",
        "    )\n",
        "\n",
        "    if add_python_path and python_path not in sys.path:\n",
        "        logger.info(\"add {} to PYTHONPATH\".format(python_path))\n",
        "        sys.path.append(python_path)\n",
        "\n",
        "    if os.path.isdir(os.path.join(python_path, \"rdkit\")):\n",
        "        logger.info(\"rdkit is already installed\")\n",
        "        if not force:\n",
        "            return\n",
        "\n",
        "        logger.info(\"force re-install\")\n",
        "\n",
        "    url = url_base + file_name\n",
        "    python_version = \"{0}.{1}.{2}\".format(*sys.version_info)\n",
        "\n",
        "    logger.info(\"python version: {}\".format(python_version))\n",
        "\n",
        "    if os.path.isdir(conda_path):\n",
        "        logger.warning(\"remove current miniconda\")\n",
        "        shutil.rmtree(conda_path)\n",
        "    elif os.path.isfile(conda_path):\n",
        "        logger.warning(\"remove {}\".format(conda_path))\n",
        "        os.remove(conda_path)\n",
        "\n",
        "    logger.info('fetching installer from {}'.format(url))\n",
        "    res = requests.get(url, stream=True)\n",
        "    res.raise_for_status()\n",
        "    with open(file_name, 'wb') as f:\n",
        "        for chunk in res.iter_content(chunk_size):\n",
        "            f.write(chunk)\n",
        "    logger.info('done')\n",
        "\n",
        "    logger.info('installing miniconda to {}'.format(conda_path))\n",
        "    subprocess.check_call([\"bash\", file_name, \"-b\", \"-p\", conda_path])\n",
        "    logger.info('done')\n",
        "\n",
        "    logger.info(\"installing rdkit\")\n",
        "    subprocess.check_call([\n",
        "        os.path.join(conda_path, \"bin\", \"conda\"),\n",
        "        \"install\",\n",
        "        \"--yes\",\n",
        "        \"-c\", \"rdkit\",\n",
        "        \"python=={}\".format(python_version),\n",
        "        \"rdkit\" if rdkit_version is None else \"rdkit=={}\".format(rdkit_version)])\n",
        "    logger.info(\"done\")\n",
        "\n",
        "    import rdkit\n",
        "    logger.info(\"rdkit-{} installation finished!\".format(rdkit.__version__))\n",
        "\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    install()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "OVlE5Toqx_CQ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from rdkit import Chem\n",
        "generatedMol = [\"PCPPOO\",\"PPPCPCPO\", \"Cl.CNCN1C2CCCC2CCC2CCCCC12\", \"CCCPCPCO\", \"CNPCP=O\", \"CC1=C(CC(=O)O)c2cc(Cl)ccc2/C/1=C\\c1ccc(cc1)F\", \"NC(=O)c1ccc(I)c(c1)F\", \"PPNCCCCPO\", \"CN1CCCC1c2cccnc2\", \"CC(C)[C@H](N)C(=O)N1CCC[C@H]1S(O)O\"]\n",
        "inputMols = []\n",
        "for smiles in generatedMol:\n",
        "  inputMols.append(Chem.MolFromSmiles(smiles))\n",
        "generatedMol = inputMols\n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Io43_XqiNTyf",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "generatedMol"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LbNNcAK3fyv-",
        "colab_type": "text"
      },
      "source": [
        "Changes in molecules: 3, 7(deleted)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "56d-SUwp5U7d",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!pip install boost-py"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RNghz1lbyq-N",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "n = 10\n",
        "from rdkit.Chem import Descriptors\n",
        "properties = [\"Mol_weight\",\"FpDensityMorgan1\",\"MaxAbsPartialCharge\",\"NumHeavyAtoms\",\"NumRotatableBonds\",\"NumAromaticRings\", \"NumHBA\", \"NumHBD\", \"NumLipinskiHBA\", \"NumLipinskiHBD\", \"TPSA\", \"QED_CALC\", \"LOGP\"]\n",
        "propArr = {}\n",
        "for factor in properties:\n",
        "  propArr[factor] = []\n",
        "value = 0 \n",
        "for i in range(0,n):\n",
        "  for factor in properties:\n",
        "    if factor == \"Mol_weight\":\n",
        "      value = Descriptors.ExactMolWt(generatedMol[i])\n",
        "      propArr[factor].append(value)\n",
        "    if factor == \"FpDensityMorgan1\":\n",
        "      value = Descriptors.FpDensityMorgan1(generatedMol[i])\n",
        "      propArr[factor].append(value)\n",
        "    if factor == \"MaxAbsPartialCharge\":\n",
        "      value = Descriptors.MaxAbsPartialCharge(generatedMol[i])\n",
        "      propArr[factor].append(value)\n",
        "    if factor == \"NumHeavyAtoms\":\n",
        "      value = Chem.Lipinski.HeavyAtomCount(generatedMol[i])\n",
        "      propArr[factor].append(value)\n",
        "    if factor == \"NumRotatableBonds\":\n",
        "      value = Chem.rdMolDescriptors.CalcNumRotatableBonds(generatedMol[i])\n",
        "      propArr[factor].append(value)\n",
        "    if factor == \"NumAromaticRings\":\n",
        "      value = Chem.rdMolDescriptors.CalcNumAromaticRings(generatedMol[i])\n",
        "      propArr[factor].append(value)\n",
        "    if factor == \"NumHBA\":\n",
        "      value = Chem.rdMolDescriptors.CalcNumHBA(generatedMol[i])\n",
        "      propArr[factor].append(value)\n",
        "    if factor == \"NumHBD\":\n",
        "      value = Chem.rdMolDescriptors.CalcNumHBD(generatedMol[i])\n",
        "      propArr[factor].append(value) \n",
        "    if factor == \"NumLipinskiHBA\":\n",
        "      value = Chem.rdMolDescriptors.CalcNumLipinskiHBA(generatedMol[i])\n",
        "      propArr[factor].append(value)\n",
        "    if factor == \"NumLipinskiHBD\":\n",
        "      value = Chem.rdMolDescriptors.CalcNumLipinskiHBD(generatedMol[i])\n",
        "      propArr[factor].append(value)\n",
        "    if factor == \"TPSA\":\n",
        "      value = Chem.rdMolDescriptors.CalcTPSA(generatedMol[i])\n",
        "      propArr[factor].append(value) \n",
        "    if factor == \"QED_CALC\":\n",
        "      value = Chem.QED.weights_mean(generatedMol[i])\n",
        "      propArr[factor].append(value) \n",
        "    if factor == \"LOGP\":\n",
        "      value = Chem.Crippen.MolLogP(generatedMol[i])\n",
        "      propArr[factor].append(value) \n",
        "      \n",
        "  "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2ZO9hu8E8BFV",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "for factor in properties:\n",
        "  print(propArr[factor])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rdg97M4o-cFM",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "index1 = []\n",
        "for i in range(1,11):\n",
        "    index1.append(i)\n",
        "\n",
        "data = pd.DataFrame(data = propArr, index = index1)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "b3VxG6_yrFhX",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#LinpinskiRuleCheck\n",
        "check = [0]*10\n",
        "\n",
        "for i in range(0,10):\n",
        "  if propArr[\"Mol_weight\"][i] > 500:\n",
        "    check[i] += 1\n",
        "  if propArr[\"NumLipinskiHBA\"][i] > 10:\n",
        "    check[i] += 1\n",
        "  if propArr[\"NumLipinskiHBD\"][i] > 5:\n",
        "    check[i] += 1\n",
        "  if propArr[\"LOGP\"][i] > 5:\n",
        "    check[i] += 1\n",
        "    "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HtrFCr-V9yH0",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#GeneralRo5Check\n",
        "\n",
        "check0 = [0]*10\n",
        "\n",
        "for i in range(0,10):\n",
        "  if propArr[\"Mol_weight\"][i] > 500:\n",
        "    check[i] += 1\n",
        "  if propArr[\"NumHBA\"][i] > 10:\n",
        "    check[i] += 1\n",
        "  if propArr[\"NumHBD\"][i] > 5:\n",
        "    check[i] += 1\n",
        "  if propArr[\"LOGP\"][i] > 5:\n",
        "    check[i] += 1"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "n7p6mtJ12iyU",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#ro3 check\n",
        "\n",
        "check1 = [1]*10\n",
        "\n",
        "for i in range(0,10):\n",
        "  if propArr[\"Mol_weight\"][i] > 300:\n",
        "    check1[i] = 0\n",
        "    break\n",
        "  if propArr[\"NumHBA\"][i] > 3:\n",
        "    check1[i] = 0\n",
        "    break\n",
        "  if propArr[\"NumHBD\"][i] > 3:\n",
        "    check1[i] = 0\n",
        "    break\n",
        "  if propArr[\"LOGP\"][i] > 3:\n",
        "    check1[i] = 0\n",
        "    break\n",
        "    \n",
        " "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2dEfUCn55tpk",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#adding ro5 and ro3 results\n",
        "\n",
        "addProps = [\"num_lipinski_ro5_violations\",\"num_ro5_violations\", \"ro3_pass\"]\n",
        "\n",
        "for prop in addProps:\n",
        "  propArr[prop] = []\n",
        "value = 0\n",
        "for i in range(0,n):\n",
        "  \n",
        "    value = check[i]\n",
        "    propArr[\"num_lipinski_ro5_violations\"].append(value)\n",
        "    value = check0[i]\n",
        "    propArr[\"num_ro5_violations\"].append(value)\n",
        "    value = check1[i]\n",
        "    propArr[\"ro3_pass\"].append(value)\n",
        "    \n",
        "    \n",
        "    \n",
        "   \n",
        "    \n",
        "\n",
        "                       \n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "g6OpRVlJA-x0",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "propArr"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5DRAEQuc4J-0",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BV5x3rfCUYHb",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!pip install chembl_webresource_client"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BzrGUNw0BoVd",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "\n",
        "drug_indication = new_client.drug_indication\n",
        "molecules = new_client.molecule\n",
        "lung_cancer_ind = drug_indication.filter(efo_term__icontains=\"LUNG CARCINOMA\")\n",
        "lung_cancer_mols = molecules.filter(molecule_chembl_id__in=[x['molecule_chembl_id'] for x in lung_cancer_ind])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nB97-zVF5aGA",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "lung_cancer_mols"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Tjz8VHWQZkrJ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "\n",
        "\n",
        "\n",
        "    \n",
        "arr = lung_cancer_mols[0]['molecule_properties']\n",
        "\n",
        "arr['active'] = 1\n",
        "arr_labels = []\n",
        "for key in arr.keys():\n",
        "    arr_labels.append(key)\n",
        "a = []\n",
        "for i in range(23):\n",
        "    a.append([])   \n",
        "    \n",
        "\n",
        "\n",
        "for q in range(0,23):\n",
        "    a[q].append(arr[arr_labels[q]])\n",
        "    \n",
        "    \n",
        "\n",
        "\n",
        "index = []\n",
        "for p in range(1,50):\n",
        "    count = 1\n",
        "    list1 = lung_cancer_mols[p]['molecule_properties']\n",
        "    \n",
        "    if list1 != None:\n",
        "      list1['active'] = 1\n",
        "      if lung_cancer_mols[p]['molecule_properties'] != None and lung_cancer_mols[p]['molecule_properties']['alogp'] != None and lung_cancer_mols[p]['molecule_properties']['full_mwt'] != None:\n",
        "        if float(lung_cancer_mols[p]['molecule_properties']['full_mwt']) <= float(300):\n",
        "\n",
        "          for j in range(0,23):\n",
        "              a[j].append(list1[arr_labels[j]])\n",
        "          count += 1\n",
        "\n",
        "l1 = len(a[0])\n",
        "def makeDict(list1, list2):\n",
        "    new_Dict = {}\n",
        "    for i in range(0,23):\n",
        "        new_Dict[list1[i]] = list2[i]\n",
        "    return new_Dict\n",
        "for i in range(1,l1+1):\n",
        "    index.append(i)\n",
        "\n",
        "data2 = pd.DataFrame(data = makeDict(arr_labels, a),index = index, copy = True)\n",
        "data2 = data2[arr_labels]\n",
        "data2.head()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fmzmMSuSSDxn",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "orig = len(data2)\n",
        "\n",
        "\n",
        "\n",
        "y = []\n",
        "for i in range (0,l1):\n",
        "  y.append([])\n",
        "predProps = {'acd_logd','acd_logp', 'mw_monoisotopic', 'mw_freebase'}\n",
        "data2['acd_logd'] = data2['acd_logd'].apply(pd.to_numeric, errors = 'ignore')\n",
        "data2['acd_logp'] = data2['acd_logp'].apply(pd.to_numeric, errors = 'ignore')\n",
        "data2['mw_monoisotopic'] = data2['mw_monoisotopic'].apply(pd.to_numeric, errors = 'ignore')\n",
        "data2['mw_freebase'] = data2['mw_freebase'].apply(pd.to_numeric, errors = 'ignore')\n",
        "y[0] = data2['active'].values\n",
        "y[1] = data2['acd_logd'].values\n",
        "y[2] = data2['acd_logp'].values\n",
        "y[3] = data2['mw_freebase'].values\n",
        "y[4] = data2['mw_monoisotopic'].values\n",
        "\n",
        "\n",
        "    \n",
        "data2.drop(labels = ['active', 'acd_logd','acd_logp','mw_monoisotopic','mw_freebase','molecular_species', 'acd_most_apka', 'acd_most_bpka', 'full_molformula'], axis = 1, inplace = True)\n",
        "data2['ro3_pass'].replace(to_replace=['Y'], value = 1,inplace=True)\n",
        "data2['ro3_pass'].replace(to_replace = ['N'], value = 0, inplace=True)\n",
        "data2 = data2.apply(pd.to_numeric, errors = 'ignore')\n",
        "X = data2.iloc[::].values\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "f5dawXgoVc88",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y[1][4]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GTyA6kGZFkgf",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "X"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "x0uI2-xVe2Zv",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from statistics import median\n",
        "import numpy as np\n",
        "import math\n",
        "for i in range(0,orig):\n",
        "    for j in range(0, len(X[i])):\n",
        "        lst = []\n",
        "        if math.isnan(X[i][j]):\n",
        "            for b in range(0,orig):\n",
        "                if math.isnan(X[b][j]) == False:\n",
        "                    \n",
        "                    lst.append(X[b][j])\n",
        "            X[i][j] = median(lst)\n",
        "            \n",
        "for i in range(0,orig):\n",
        "    for j in range(0, len(X[i])):\n",
        "        X[i][j] = np.float32(X[i][j])\n",
        "def median(lst):\n",
        "    sortedLst = sorted(lst)\n",
        "    lstLen = len(lst)\n",
        "    index = (lstLen - 1) // 2\n",
        "\n",
        "    if (lstLen % 2):\n",
        "        return sortedLst[index]\n",
        "    else:\n",
        "        return (sortedLst[index] + sortedLst[index + 1])/2.0\n",
        " "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NMOImnK3ypWv",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "data3 = {}\n",
        "\n",
        "\n",
        "\n",
        "for i in range(0,n):\n",
        "  if i == 0:\n",
        "    data3['alogp'] = []\n",
        "    data3['aromatic_rings'] = []\n",
        "    data3['full_mwt'] = []\n",
        "    data3['hba'] = []\n",
        "    data3['hba_lipinski'] = []\n",
        "    data3['hbd'] = []\n",
        "    data3['hbd_lipinski'] = []\n",
        "    data3['heavy_atoms'] = []\n",
        "    data3['num_lipinski_ro5_violations'] = []\n",
        "    data3['num_ro5_violations'] = []\n",
        "    data3['psa'] = []\n",
        "    data3['qed_weighted'] = [] \n",
        "    data3['ro3_pass'] = []\n",
        "    data3['rtb'] = []\n",
        "    \n",
        "  data3['alogp'].append(propArr[\"LOGP\"][i])\n",
        "  data3['aromatic_rings'].append(propArr[\"NumAromaticRings\"][i])\n",
        "  data3['full_mwt'].append(propArr[\"Mol_weight\"][i])\n",
        "  data3['hba'].append(propArr[\"NumHBA\"][i])\n",
        "  data3['hba_lipinski'].append(propArr[\"NumLipinskiHBA\"][i])\n",
        "  data3['hbd'].append(propArr[\"NumHBD\"][i])\n",
        "  data3['hbd_lipinski'].append(propArr[\"NumLipinskiHBD\"][i])\n",
        "  data3['heavy_atoms'].append(propArr[\"NumHeavyAtoms\"][i])\n",
        "  data3['num_lipinski_ro5_violations'].append(propArr[\"num_lipinski_ro5_violations\"][i])\n",
        "  data3['num_ro5_violations'].append(propArr[\"num_ro5_violations\"][i])\n",
        "  data3['psa'].append(propArr[\"TPSA\"][i])\n",
        "  data3['qed_weighted'].append(propArr[\"QED_CALC\"][i])\n",
        "  data3['ro3_pass'].append(propArr['ro3_pass'][i])\n",
        "  data3['rtb'].append(propArr['NumRotatableBonds'][i])\n",
        "  \n",
        "index = []\n",
        "for q in range(1,11):\n",
        "  index.append(q)\n",
        "      \n",
        "X_test1 = pd.DataFrame(data = data3, index = index) \n",
        "X_test1 = X_test1.apply(pd.to_numeric, errors = 'ignore')\n",
        "X_test1.head()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QSkDaajoe7g8",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.model_selection import train_test_split\n",
        "#Y will be used to collect the calculated properties for each of the 10 compounds\n",
        "Y = {}\n",
        "predProps2 = ['acd_logd','acd_logp', 'mw_monoisotopic', 'mw_freebase']\n",
        "acc = []\n",
        "for i in range(0,4):\n",
        "  #division of main dataset\n",
        "  X_train, X_test, y_train, y_test = train_test_split(X, y[i+1], test_size = 0.3, random_state = 0)\n",
        "  rf = RandomForestRegressor(n_estimators = 50)\n",
        "  \n",
        "  #Fitting the training datasets\n",
        "  rf.fit(X_train,y_train)\n",
        "  #Predicted values\n",
        "  pred = rf.predict(X_test)\n",
        "  counter = 0 \n",
        "  for j in range(0, len(pred)):\n",
        "    #accuracy\n",
        "    if pred[j] >= 0.9*y_test[j] and pred[j] <= 1.1*(y_test[j]):\n",
        "      counter += 1\n",
        "  print(counter/len(pred)*100)\n",
        "  #properties for new generated compounds\n",
        "  Y_test1 = rf.predict(X_test1)\n",
        "  Y[predProps2[i]] = Y_test1"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GaWkQfV-Nq6j",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "for predProps in predProps2:\n",
        "  print(Y[predProps])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PJzMb7eh0IFJ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "for i in range(0,n):\n",
        "  if i == 0:\n",
        "    data3['mw_freebase'] = []\n",
        "    data3['mw_monoisotopic'] = []\n",
        "    \n",
        "  data3['mw_freebase'].append(Y[predProps2[2]][i])\n",
        "  data3['mw_monoisotopic'].append(Y[predProps2[3]][i])\n",
        "  \n",
        "  \n",
        "X_test1 = pd.DataFrame(data = data3, index = index)\n",
        "X_test1 = X_test1.apply(pd.to_numeric, errors = 'ignore')\n",
        "cols = X_test1.columns.tolist()\n",
        "temp = cols[14]\n",
        "cols[14] = cols[15]\n",
        "cols [15] = temp\n",
        "print(cols)\n",
        "cols = cols[0:8] + cols[-2:] + cols [8:(len(cols)-2)]\n",
        "X_test1 = X_test1[cols]\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eOCOezVuWZj2",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "X_test1.head()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JyP3FjQnHX9N",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "X_test2 = X_test1.iloc[::].values"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kt6TcNFjfPxl",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "\n",
        "\n",
        "\n",
        "    \n",
        "arr = lung_cancer_mols[0]['molecule_properties']\n",
        "\n",
        "arr['active'] = 1\n",
        "arr_labels = []\n",
        "for key in arr.keys():\n",
        "    arr_labels.append(key)\n",
        "a = []\n",
        "for i in range(23):\n",
        "    a.append([])   \n",
        "    \n",
        "\n",
        "\n",
        "for q in range(0,23):\n",
        "    a[q].append(arr[arr_labels[q]])\n",
        "    \n",
        "    \n",
        "\n",
        "\n",
        "index = []\n",
        "for p in range(1,50):\n",
        "    count = 1\n",
        "    list1 = lung_cancer_mols[p]['molecule_properties']\n",
        "    \n",
        "    if list1 != None:\n",
        "      list1['active'] = 1\n",
        "      if lung_cancer_mols[p]['molecule_properties'] != None and lung_cancer_mols[p]['molecule_properties']['alogp'] != None and lung_cancer_mols[p]['molecule_properties']['full_mwt'] != None:\n",
        "        if float(lung_cancer_mols[p]['molecule_properties']['full_mwt']) <= float(350):\n",
        "          for j in range(0,23):\n",
        "              a[j].append(list1[arr_labels[j]])\n",
        "          count += 1\n",
        "\n",
        "l1 = len(a[0])\n",
        "def makeDict(list1, list2):\n",
        "    new_Dict = {}\n",
        "    for i in range(0,23):\n",
        "        new_Dict[list1[i]] = list2[i]\n",
        "    return new_Dict\n",
        "for i in range(1,l1+1):\n",
        "    index.append(i)\n",
        "    \n",
        "dataLung = pd.DataFrame(data = makeDict(arr_labels, a),index = index, copy = True)\n",
        "dataLung = dataLung[arr_labels]\n",
        "dataLung.head()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Gp0RHFxEdb6Y",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "\n",
        "drug_indication = new_client.drug_indication\n",
        "molecules = new_client.molecule\n",
        "ovary_cancer_ind = drug_indication.filter(efo_term__icontains=\"OVARIAN NEOPLASM\")\n",
        "ovary_cancer_mols = molecules.filter(molecule_chembl_id__in=[x['molecule_chembl_id'] for x in ovary_cancer_ind])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PE38JnJFdryE",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "drug_indication = new_client.drug_indication\n",
        "molecules = new_client.molecule\n",
        "pancreatic_cancer_ind = drug_indication.filter(efo_term__icontains=\"PANCREATIC NEOPLASM\")\n",
        "pancreatic_cancer_mols = molecules.filter(molecule_chembl_id__in=[x['molecule_chembl_id'] for x in pancreatic_cancer_ind])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6U1kasq8pn9r",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#tb\n",
        "drug_indication = new_client.drug_indication\n",
        "molecules = new_client.molecule\n",
        "tb_ind = drug_indication.filter(efo_term__icontains=\"TUBERCULOSIS\")\n",
        "tb_mols = molecules.filter(molecule_chembl_id__in=[x['molecule_chembl_id'] for x in tb_ind])\n",
        "\n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IeGUDKtFf2mS",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "pancreatic_cancer_mols[0]['molecule_properties']"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8S4TV74Te0mK",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "\n",
        "\n",
        "\n",
        "    \n",
        "arr = pancreatic_cancer_mols[0]['molecule_properties']\n",
        "\n",
        "\n",
        "arr['active'] = 0\n",
        "arr_labels = []\n",
        "for key in arr.keys():\n",
        "    arr_labels.append(key)\n",
        "a = []\n",
        "for i in range(23):\n",
        "    a.append([])   \n",
        "    \n",
        "\n",
        "\n",
        "for q in range(0,23):\n",
        "    a[q].append(arr[arr_labels[q]])\n",
        "    \n",
        "\n",
        "\n",
        "index2 = []\n",
        "for p in range(1,30):\n",
        "    count = 1\n",
        "    list1 = pancreatic_cancer_mols[p]['molecule_properties']\n",
        "    if list1 != None:\n",
        "      list1['active'] = 0\n",
        "      if pancreatic_cancer_mols[p]['molecule_properties'] != None and pancreatic_cancer_mols[p]['molecule_properties']['alogp'] != None and pancreatic_cancer_mols[p]['molecule_properties']['full_mwt'] != None:\n",
        "        if float(pancreatic_cancer_mols[p]['molecule_properties']['full_mwt']) <= float(300):\n",
        "          for j in range(0,23):\n",
        "              a[j].append(list1[arr_labels[j]])\n",
        "          count += 1\n",
        "\n",
        "l2 = len(a[0])\n",
        "def makeDict(list1, list2):\n",
        "    new_Dict = {}\n",
        "    for i in range(0,23):\n",
        "        new_Dict[list1[i]] = list2[i]\n",
        "    return new_Dict\n",
        "\n",
        "for i in range(l1+1, l1 + l2 + 1):\n",
        "    index2.append(i)\n",
        "    \n",
        "dataPan = pd.DataFrame(data = makeDict(arr_labels, a), index = index2)\n",
        "dataPan = dataPan[arr_labels]\n",
        "dataPan.head()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qa1FUQP5rhox",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "\n",
        "\n",
        "\n",
        "    \n",
        "arr = tb_mols[0]['molecule_properties']\n",
        "\n",
        "\n",
        "arr['active'] = 0\n",
        "arr_labels = []\n",
        "for key in arr.keys():\n",
        "    arr_labels.append(key)\n",
        "a = []\n",
        "for i in range(23):\n",
        "    a.append([])   \n",
        "    \n",
        "\n",
        "\n",
        "for q in range(0,23):\n",
        "    a[q].append(arr[arr_labels[q]])\n",
        "    \n",
        "\n",
        "\n",
        "index2 = []\n",
        "for p in range(1,40):\n",
        "    count = 1\n",
        "    list1 = tb_mols[p]['molecule_properties']\n",
        "    if list1 != None:\n",
        "      list1['active'] = 0\n",
        "      if list1 != None and list1['alogp'] != None and list1['full_mwt'] != None:\n",
        "        if float(list1['full_mwt']) <= float(400):\n",
        "          for j in range(0,23):\n",
        "              a[j].append(list1[arr_labels[j]])\n",
        "          count += 1\n",
        "\n",
        "l2 = len(a[0])\n",
        "def makeDict(list1, list2):\n",
        "    new_Dict = {}\n",
        "    for i in range(0,23):\n",
        "        new_Dict[list1[i]] = list2[i]\n",
        "    return new_Dict\n",
        "\n",
        "for i in range(l1+1, l1 + l2 + 1):\n",
        "    index2.append(i)\n",
        "    \n",
        "datatb = pd.DataFrame(data = makeDict(arr_labels, a), index = index2)\n",
        "datatb = datatb[arr_labels]\n",
        "datatb.head()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "o6G9Bbi0c5RA",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "\n",
        "\n",
        "\n",
        "    \n",
        "arr = ovary_cancer_mols[0]['molecule_properties']\n",
        "\n",
        "\n",
        "arr['active'] = 0\n",
        "arr_labels = []\n",
        "for key in arr.keys():\n",
        "    arr_labels.append(key)\n",
        "a = []\n",
        "for i in range(23):\n",
        "    a.append([])   \n",
        "    \n",
        "\n",
        "\n",
        "for q in range(0,23):\n",
        "    a[q].append(arr[arr_labels[q]])\n",
        "    \n",
        "\n",
        "\n",
        "index2 = []\n",
        "for p in range(1,30):\n",
        "    count = 1\n",
        "    list1 = ovary_cancer_mols[p]['molecule_properties']\n",
        "    if list1 != None:\n",
        "      list1['active'] = 0\n",
        "      if ovary_cancer_mols[p]['molecule_properties'] != None and ovary_cancer_mols[p]['molecule_properties']['alogp'] != None and ovary_cancer_mols[p]['molecule_properties']['full_mwt'] != None:\n",
        "        if float(ovary_cancer_mols[p]['molecule_properties']['full_mwt']) <= float(300):\n",
        "          for j in range(0,23):\n",
        "              a[j].append(list1[arr_labels[j]])\n",
        "          count += 1\n",
        "\n",
        "l3 = len(a[0])\n",
        "def makeDict(list1, list2):\n",
        "    new_Dict = {}\n",
        "    for i in range(0,23):\n",
        "        new_Dict[list1[i]] = list2[i]\n",
        "    return new_Dict\n",
        "\n",
        "for i in range(l1 + l2 +1, l1 + l2 + +l3 + 1):\n",
        "    index2.append(i)\n",
        "    \n",
        "dataOv = pd.DataFrame(data = makeDict(arr_labels, a), index = index2)\n",
        "dataOv = dataOv[arr_labels]\n",
        "dataOv.head()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vcU6_dIEflxL",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aUqC4axod5zw",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "frames = [dataLung, dataOv, dataPan]\n",
        "\n",
        "dataComb = pd.concat(frames)\n",
        "\n",
        "dataComb.head()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Qyo3ciS6tDo5",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "frames = [dataLung, datatb]\n",
        "\n",
        "dataComb = pd.concat(frames)\n",
        "\n",
        "dataComb.head()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "z8pHqSQ-f2x1",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "orig = len(dataComb)\n",
        "\n",
        "dataComb.drop(labels = ['full_molformula', \"acd_most_apka\",\"acd_most_bpka\",\"molecular_species\",\"acd_logd\",\"acd_logp\"],axis = 1,inplace= True)\n",
        "\n",
        "dataComb['ro3_pass'].replace(to_replace=['Y'], value = 1,inplace=True)\n",
        "dataComb['ro3_pass'].replace(to_replace = ['N'], value = 0, inplace=True)\n",
        "\n",
        "dataComb = dataComb.apply(pd.to_numeric, errors = 'ignore')\n",
        "\n",
        "y = dataComb['active'].values\n",
        "\n",
        "dataComb.drop(labels = ['active'],axis = 1,inplace= True)\n",
        "X = dataComb.iloc[::].values"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6fO3qLfHdkw3",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from statistics import median\n",
        "import numpy as np\n",
        "import math\n",
        "for i in range(0,l2 +l1):\n",
        "    for j in range(0, len(X[i])):\n",
        "        lst = []\n",
        "        if math.isnan(X[i][j]):\n",
        "            for b in range(0,l2+l1):\n",
        "                if math.isnan(X[b][j]) == False:\n",
        "                    \n",
        "                    lst.append(X[b][j])\n",
        "            X[i][j] = median(lst)\n",
        "            \n",
        "for i in range(0,l2+l1):\n",
        "    for j in range(0, len(X[i])):\n",
        "        X[i][j] = np.float32(X[i][j])\n",
        "def median(lst):\n",
        "    sortedLst = sorted(lst)\n",
        "    lstLen = len(lst)\n",
        "    index = (lstLen - 1) // 2\n",
        "\n",
        "    if (lstLen % 2):\n",
        "        return sortedLst[index]\n",
        "    else:\n",
        "        return (sortedLst[index] + sortedLst[index + 1])/2.0"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "r1Kd94ujkteH",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#model = random forest classifier\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "rf = RandomForestClassifier(n_estimators = 200)\n",
        "rf.fit(X_train,y_train)\n",
        "pred1 = rf.predict(X_test)\n",
        "#accuracy\n",
        "from sklearn.metrics import accuracy_score\n",
        "score = accuracy_score(y_test,pred1)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "r1YK2ph4iOYW",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "X_test1.head()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IujMDUUW_HDW",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "pred2 = rf.predict(X_test1)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-j2aFafgB9ZL",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "pred2"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BhXBvSZW3DMw",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "score"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "f8l1gKs_pIEb",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#evolutionary algo approach\n",
        "\n",
        "!pip install deap update_checker tqdm\n",
        "!pip install tpot"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "X-kp2mxBpKrt",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import numpy as np \n",
        "import pandas as pd \n",
        "import matplotlib.pyplot as plt \n",
        "%matplotlib inline \n",
        "from sklearn import preprocessing \n",
        "from sklearn.metrics import mean_squared_error "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0yGKodUkpS5P",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#a final properties array\n",
        "\n",
        "props = [\"alogp\" ,\"aromatic_rings\", \"full_molformula\",\"full_mwt\", \"hba\", \"hba_lipinski\",\"hbd\", \"hbd_lipinski\",'heavy_atoms', \"mw_freebase\",\"mw_monoistopic\",\"num_lipinski_ro5_violations\", \"num_ro5_violations\",\"psa\",\"qed_weighted\", \"ro3_pass\",\"rtb\"]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8J94NhXmpbcl",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# finally building model using tpot library\n",
        "from tpot import TPOTRegressor\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y,\n",
        " train_size=0.75, test_size=0.25)\n",
        "\n",
        "tpot = TPOTRegressor(generations=12, population_size=50, verbosity=2)\n",
        "tpot.fit(X_train, y_train)\n",
        "print(tpot.score(X_test, y_test))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "N7OTJfUPck7Q",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "tpot_pred = tpot.predict(X_test)\n",
        "for i in range(0,len(tpot_pred)):\n",
        "  tpot_pred[i] = int(round(tpot_pred[i]))\n",
        "  \n",
        "tpot_pred"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GZ4BqTP4c2vb",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y_test"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jw27G5y9pzVC",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#my pred\n",
        "\n",
        "tpot_pred1 = tpot.predict(X_test1)\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8mUXIW54qtiT",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "tpot_pred1"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QBH6HoXZ4q-v",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!pip install tpot"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nd6xzUFIdLas",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# finally building model using tpot library\n",
        "from tpot import TPOTClassifier\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y,\n",
        " train_size=0.75, test_size=0.25)\n",
        "\n",
        "tpot = TPOTClassifier(generations=10, population_size=100, verbosity=2)\n",
        "tpot.fit(X_train, y_train)\n",
        "print(tpot.score(X_test, y_test))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xecnB6pi1W_w",
        "colab_type": "text"
      },
      "source": [
        "GradientBoostingClassifier = 0.6"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ztR-qTf5fKhM",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "tpot_testpred = tpot.predict(X_test)\n",
        "tpot_testpred"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Zy-kluBZfXAn",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y_test"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pSp5qam5gO9r",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from sklearn.metrics import accuracy_score"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EAWOGWqBfe6H",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "score = accuracy_score(y_test,tpot_testpred)\n",
        "score"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6OOHIkQHdMp7",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "tpot_pred2 = tpot.predict(X_test1)\n",
        "tpot_pred2"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}