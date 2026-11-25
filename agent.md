# Contexto do Projeto: Reprodução e Intervenção do DeepCache (CVPR 2024)

## Objetivo
1. Reproduzir o baseline do artigo "DeepCache: Accelerating Diffusion Models for Free" (Ma et al., CVPR 2024) usando Stable Diffusion 1.5 e 2.1.
2. Implementar uma intervenção tencica para melhorar a eficiência do DeepCache, explorando técnicas de cache mais avançadas e estratégias de otimização, isolada na politica de atualização de cache da U-Net.
3. Coletar metricas comparativas (tempo de inferência, uso de memória, qualidade da imagem gerada) entre o baseline e a intervenção proposta.

## Regras de Engenharia para o Agente
- **Isolamento:** A intervenção deve ser isolada na política de atualização de cache da U-Net, sem modificar outras partes do modelo.
- **Reprodutibilidade:** O código deve ser estruturado de forma que outros pesquisadores possam reproduzir os resultados facilmente, com documentação clara e instruções de execução.
- **Comparabilidade:** As métricas coletadas devem ser consistentes e comparáveis entre o baseline e a intervenção, utilizando os mesmos conjuntos de dados e condições de teste.
- **Eficiência:** A intervenção deve visar melhorias significativas em termos de tempo de inferência e uso de memória, sem comprometer a qualidade da imagem gerada.
- **Documentação:** Todo o processo de implementação, desde a configuração do ambiente até a execução dos experimentos, deve ser bem documentado, incluindo detalhes sobre as alterações feitas na política de cache e os resultados obtidos.