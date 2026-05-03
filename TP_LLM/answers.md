## **2. Exploration du fonctionnement du LLM**

### **2.1 Tokenizer**

**Remarques sur les différents tokenizers (ex: Qwen vs GPT-5.X) :**

- Les tokenizers ne découpent pas les mots de la même manière. Un même texte donnera une séquence de tokens différente selon le modèle.
- Les modèles plus récents ou plus gros possèdent souvent un vocabulaire plus large. Par conséquent, ils arrivent à "compresser" un même texte en utilisant moins de tokens.
- Ils gèrent également beaucoup mieux les langues autres que l'anglais (français, chinois, etc.), en évitant de découper des mots en plusieurs sous-tokens.

**Conséquences sur le traitement et la génération :**

- **Qualité :** Un prompt mieux découpé (avec 1 token = 1 mot) sera mieux interprété par le LLM qui n'aura pas besoin de comprendre et relier chaque sous-tokens de chaque mot.
- **Performance :** Moins il y a de tokens pour un même mot, plus la génération sera rapide (le LLM générant séquentiellement token par token) et moins elle sera coûteuse (voir la suite du TP).

### **2.2 LLM : Embedding**

**Limites de cette représentation :**

- La première couche d'embedding est **statique**. Cela signifie qu'un mot (ou sous-mot) est toujours représenté par le même vecteur numérique, quel que soit le contexte de la phrase. Par exemple, le token "avocat" aura le même vecteur qu'il s'agisse du fruit ou du métier, ce qui crée une ambiguïté sémantique. Les relations complexes (négation, ironie, dépendances longues) ne sont pas représentées dans ces vecteurs initiaux.

**Utilité du réseau de neurones (Transformer) :**

- C'est ici qu'interviennent les milliards de paramètres ! L'architecture Transformer va **contextualiser** ces vecteurs statiques. Le réseau va modifier le vecteur de chaque token en observant les tokens environnants. Ainsi, à la sortie du réseau, le vecteur représentant "avocat" aura été rapproché du champ lexical de la justice si le mot "tribunal" est dans le prompt, résolvant ainsi l'ambiguïté.

### **2.3 LLM : Réseau de neurones**

**Influence de la `temperature` (entre 0 et 1) :**

- La température lisse ou accentue les écarts de probabilité entre les tokens.
- **Proche de 0 :** La distribution devient très "pointue". Le modèle est presque déterministe et choisira systématiquement le token avec la probabilité maximale, écrasant les autres.
- **Proche de 1 :** La distribution est lissée. Les tokens moins probables regagnent des chances d'être échantillonnés, introduisant de l'aléatoire et de la créativité.

**Rôle du `top_k` et pourquoi on en a besoin :**

- Le `top_k` tronque la distribution pour ne conserver que les `k` tokens les plus probables (les autres voient leur probabilité réduite à 0).
- Le vocabulaire des LLMs contient environ 100 000 à 200 000 tokens. Même avec une bonne distribution, des tokens absurdes peuvent avoir une probabilité faible mais non nulle. Sans filtrage, ils pourraient être échantillonnés et produire du texte incohérent.

### **2.4 Pipeline de génération**

**Dans quels cas utiliser une température élevée ?**

- Pour les tâches créatives, le brainstorming, la génération d'histoires, de poèmes, ou la rédaction d'emails nécessitant des tournures de phrases variées. On privilégie la diversité.

**Dans quels cas utiliser une température faible ?**

- Pour les tâches analytiques, factuelles, la résolution de problèmes mathématiques, la traduction stricte, l'écriture de code informatique ou l'extraction de données structurées (comme du JSON). On privilégie l'exactitude et la prédictibilité.

### **2.5 Un point sur la mémoire**

**Pourquoi est-on facturé au token et pas à la question ?**

- Parce que le temps de calcul et la mémoire mobilisée sur les serveurs dépendent du nombre de tokens traités et générés. Une question d'un utilisateur peut faire 3 mots ("Qui est Napoléon ?") ou 5 pages de documentation technique à analyser. La facturation au token reflète la consommation réelle de ressources.

**Pourquoi y a-t-il une tarification différenciée entre "short context" et "long context" ?**

- Un prompt très long nécessite beaucoup plus de mémoire et de calcul qu'un prompt court. Les fournisseurs répercutent ce coût d'infrastructure massif sur les prompts très longs.

**Pourquoi y a-t-il une différence de prix entre "input tokens" et "output tokens" ?**

- **Input (Prompt) :** Les tokens d'entrée sont traités en parallèle lors d'une phase appelée _Prefill_. C'est extrêmement rapide et optimise bien l'architecture des GPU.
- **Output (Génération) :** Les tokens générés le sont de manière auto-régressive, un par un. Ce processus séquentiel monopolise la mémoire du serveur plus longtemps, ce qui coûte plus cher au fournisseur d'API.

## **3. Comment transformer un LLM en assistant ?**

**Techniques supplémentaires pour mieux faire respecter les instructions :**

- **Few-shot prompting :** Donner des exemples concrets des résultats attendus directement dans le prompt système.
- **Chain of Thought (CoT) :** Demander au modèle de "réfléchir étape par étape" avant de formuler sa réponse finale.
- **Formatage strict :** Donner un squelette JSON ou Markdown à remplir.
- **Prompt chaining :** découper une tâche complexe en étapes, chaque étape ayant son propre prompt optimisé.

**Rôle du paramètre `stop_strings` :**

- Il indique au modèle quand il doit s'arrêter de générer du texte. Sans balise de fin comme `</assistant>`, le modèle continuera d'auto-compléter indéfiniment. Il finira par simuler la suite de la conversation en s'inventant un nouveau `<user>` ou en partant dans des hallucinations.

**Impacts de l'absence de mémoire (renvoi de tout l'historique) :**

- **Sur la qualité :** Si l'historique devient trop grand, les LLMs ont tendance à subir le phénomène de _Lost in the Middle_ : ils se souviennent très bien du début du prompt (les instructions système) et de la toute fin (la dernière question), mais oublient les détails des messages situés au milieu de la conversation.
- **Sur le coût et la latence :** À chaque nouvelle question, la taille du prompt grandit (puisqu'il contient les échanges précédents). Le coût de calcul augmente proportionnellement, tout comme le temps de réponse.
- **Bonne pratique :** créer régulièrement une nouvelle conversation pour réinitialiser l'historique : ça réduit les coûts et augmente la qualité des réponses !

## **4. (Bonus) Tool Calls : vers l'IA agentique**

### **4.1 Un premier outil**

**Que se passe-t-il avec une question hors sujet (avec le prompt strict actuel) ?**

- Le modèle est "coincé" par la consigne stricte : `You must respond strictly in this exact JSON format`. Face à une question comme "Qui est le président de la France ?", il va soit forcer l'outil météo avec des arguments absurdes (ex: `{"tool": "get_tomorrow_weather", "city": "France"}`), soit ignorer la consigne et répondre en texte normal, ou générer un JSON corrompu.

**Correction du prompt système :**
On peut ajouter une condition ("Si / Alors") dans les instructions. Par exemple :
`"If the user asks about something else than weather, answer normally without JSON."` ou `"Only output JSON if you need to use the tool. Otherwise, provide a conversational answer."`

### **4.2 ReAct : la boucle complète**

Voici un exemple d'implémentation pour les outils demandés :

```python
import datetime
import json
import random
import urllib.request

# Tool 1 : random number
def get_random_number(min_val: int = 1, max_val: int = 100) -> dict:
    return {"result": random.randint(min_val, max_val)}

# Tool 2 : weekday
def get_weekday(date_str: str) -> dict:
    # Expect YYYY-MM-DD date format
    try:
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return {"weekday": date_obj.strftime("%A")}
    except ValueError:
        return {"error": "Invalid date format, expect : YYYY-MM-DD."}

# Tool 3 : ISS position
def get_ISS_position() -> dict:
    url = "http://api.open-notify.org/iss-now.json"
    with urllib.request.urlopen(url) as r:
        data = json.loads(r.read())
    return {
        "latitude": data["iss_position"]["latitude"],
        "longitude": data["iss_position"]["longitude"]
    }

# Add in tools registry
TOOLS_REGISTRY.update({
    "get_random_number": {
        "func": get_random_number,
        "description": "Returns a random number between min_val and max_val",
        "args": [{"name": "min_val", "type": "int"}, {"name": "max_val", "type": "int"}]
    },
    "get_weekday": {
        "func": get_weekday,
        "description": "Returns the weekday of a given date (YYYY-MM-DD format)",
        "args": [{"name": "date_str", "type": "str"}]
    },
    "get_ISS_position": {
        "func": get_ISS_position,
        "description": "Returns the current GPS coordinates of the International Space Station",
        "args": []
    }
})
```

**Question Bonus : L'exécution de code Python (`exec`)**

**Implémentation :**

```python
import io
import contextlib

def execute_python(code: str) -> dict:
    """Execute a Python code snippet. Use print() to produce output."""
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, {})
        output = stdout_capture.getvalue()
        return {"status": "success", "output": output or "No output"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


TOOLS_REGISTRY.update({
    "execute_python": {
        "func": execute_python,
        "description": "Execute a Python code snippet. Use print() to produce output.",
        "args": [{"name": "code", "type": "str"}]
    },
})
```

**Risques anticipés :**

- C'est une **faille de sécurité critique**. Donner à un LLM la capacité d'exécuter du code arbitraire sur la machine hôte permet, de manière intentionnelle (par _prompt injection_ d'un utilisateur malveillant) ou non (hallucination du LLM), de détruire des données (ex: `os.system('rm -rf /')`), d'exfiltrer des variables d'environnement, des clés d'API, ou d'infecter le serveur.

**Solutions possibles :**

- **Sandboxing strict :** ne jamais exécuter ce code sur la machine hôte. Il faut l'exécuter dans un environnement isolé, éphémère et jetable (comme un conteneur Docker).
- **Environnement maîtrisé :** utiliser un environnement d'execution avec des librairies connues. Ne pas autoriser l'installation de nouvelles librairies par le LLM.
- **Human-in-the-loop :** demander la validation explicite de l'utilisateur ("L'IA souhaite exécuter ce code : [...]. Autoriser ?") avant de lancer la commande `exec`.
