import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional, Sequence, Type, Union, Generator
from modules.llm.openai import OpenAILLM
from loguru import logger
from modules.types.graph import (
    Node,
    Relationship,
    GraphDocument,
    _Graph,
    UnstructuredRelation,
)

DEFAULT_NODE_TYPE = "Node"


def create_simple_model(
        cls,
        node_labels: Optional[List[str]] = None,
        rel_types: Optional[List[str]] = None,
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
    ) -> Type[_Graph]:
        pass

system_prompt = (
    "# Knowledge Graph Instructions for GPT-4\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph.\n"
    "Try to capture as much information from the text as possible without "
    "sacrificing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text.\n"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'."
    "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text.\n"
    "- **Relationships** represent connections between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "knowledge graphs. Instead of using specific and momentary types "
    "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
    "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "John Doe", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
    "always use the most complete identifier for that entity throughout the "
    'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
    "Remember, the knowledge graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)


class LLMGraphTransformer:
    def __init__(
        self,
        llm: OpenAILLM,
        allowed_nodes: List[str] = [],
        allowed_relationships: List[str] = [],
        prompt: Optional[str] = None,
        strict_mode: bool = True,
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
        max_retries: int = 3,  # 添加最大重试次数参数
    ):
        self.llm = llm
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self.prompt = prompt or self._default_prompt()
        self.node_properties = node_properties
        self.relationship_properties = relationship_properties
        self.max_retries = max_retries  # 保存最大重试次数

    def _default_prompt(self) -> str:
        return (
            "Tip: Make sure to answer in the correct format and do "
            "not include any explanations. "
            "Use the given format to extract information from the "
            "following input: {input}"
            "The output should be in JSON format like this: {format}"
        )

    def _format(self) -> str:
        return """
        {
            "nodes": [
                {   
                    "id": "Elon Musk",
                    "type": "Person",
                    "properties": {
                        "name": "Elon Musk"
                    }
                },
                {
                    "id": "OpenAI",
                    "type": "Organization",
                    "properties": {
                        "name": "OpenAI"
                    }
                }
            ],
            "relationships": [
                {
                    "source": {
                        "id": "Elon Musk",
                        "type": "Person",
                        "properties": {
                            "name": "Elon Musk"
                        }
                    },
                    "target": {
                        "id": "OpenAI",
                        "type": "Organization",
                        "properties": {
                            "name": "OpenAI"
                        }
                    },
                    "type": "suing"
                },
                {
                    "source": "OpenAI",
                    "target": "AI research",
                    "type": "focuses on"
                }
            ]
        }"""

    def process_response(self, document: str) -> GraphDocument:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": self.prompt.format(input=document, format=self._format()),
            },
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(messages)

                logger.debug(f"Response: {response}")

                if isinstance(response, Generator):
                    # 如果是流式响应，我们需要收集所有的内容
                    full_response = "".join(
                        chunk.choices[0].delta.content
                        for chunk in response
                        if chunk.choices[0].delta.content is not None
                    )
                else:
                    full_response = response.choices[0].message.content

                parsed_response = json.loads(full_response)
                break  # 如果成功解析，跳出循环
            except json.JSONDecodeError:
                # 如果JSON解析失败，则可能是包裹在Markdown的json代码块中
                if full_response.startswith("```json"):
                    logger.debug(
                        "Attempting to parse response wrapped in Markdown JSON code block"
                    )
                    full_response = full_response[7:-3]
                    logger.debug(f"Parsed response: {full_response}")
                    try:
                        parsed_response = json.loads(full_response)
                        break  # 如果成功解析，跳出循环
                    except json.JSONDecodeError:
                        if attempt == self.max_retries - 1:
                            logger.error(f"JSON parsing failed: {full_response}")
                            return GraphDocument(nodes=[], relationships=[])
                else:
                    if attempt == self.max_retries - 1:
                        logger.error(f"JSON parsing failed: {full_response}")
                        return GraphDocument(nodes=[], relationships=[])
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Error in processing response: {str(e)}")
                    return GraphDocument(nodes=[], relationships=[])

            logger.warning(f"Attempt {attempt + 1} failed, retrying...")

        nodes = [Node(**node) for node in parsed_response.get("nodes", [])]
        relationships = []
        for rel in parsed_response.get("relationships", []):
            source = rel.get("source")
            target = rel.get("target")

            # 处理 source
            if isinstance(source, str):
                source_node = next((node for node in nodes if node.id == source), None)
                if source_node is None:
                    source_node = Node(id=source, type=DEFAULT_NODE_TYPE)
                    nodes.append(source_node)
            elif isinstance(source, dict):
                source_node = Node(**source)
            else:
                continue  # 跳过无效的 source

            # 处理 target
            if isinstance(target, str):
                target_node = next((node for node in nodes if node.id == target), None)
                if target_node is None:
                    target_node = Node(id=target, type=DEFAULT_NODE_TYPE)
                    nodes.append(target_node)
            elif isinstance(target, dict):
                target_node = Node(**target)
            else:
                continue  # 跳过无效的 target

            relationships.append(
                Relationship(source=source_node, target=target_node, type=rel["type"])
            )

        if self.strict_mode:
            nodes = [node for node in nodes if node.type in self.allowed_nodes]
            relationships = [
                rel
                for rel in relationships
                if rel.type in self.allowed_relationships
                and rel.source.type in self.allowed_nodes
                and rel.target.type in self.allowed_nodes
            ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    def convert_to_graph_documents(
        self, documents: Sequence[str]
    ) -> List[GraphDocument]:
        return [self.process_response(document) for document in documents]

    @staticmethod
    def _create_knowledge_graph(graph_document: GraphDocument) -> nx.Graph:
        G = nx.Graph()
        
        # 添加节点
        for node in graph_document.nodes:
            G.add_node(node.id, type=node.type, **node.properties)
        
        # 添加边
        for rel in graph_document.relationships:
            G.add_edge(rel.source.id, rel.target.id, type=rel.type)
        
        return G
    
    @staticmethod
    def _visualize_knowledge_graph(G: nx.Graph, output_file: str = 'knowledge_graph.png'):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
        nx.draw_networkx_labels(G, pos)
        
        # 绘制边
        edge_labels = nx.get_edge_attributes(G, 'type')
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @classmethod
    def generate_knowledge_graph_image(
        cls,
        graph_documents: List[GraphDocument],
        output_path: str = 'knowledge_graph.png'
    ):
        # 合并多个GraphDocument
        combined_graph = nx.Graph()
        for doc in graph_documents:
            G = cls._create_knowledge_graph(doc)
            combined_graph = nx.compose(combined_graph, G)
        
        cls._visualize_knowledge_graph(combined_graph, output_path)
