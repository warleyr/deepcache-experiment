# Respostas do Template — Relatório Projeto: Modelos Generativos para Imagens

Este documento reproduz integralmente os campos das 4 páginas do PDF
`template-relatorio-projeto.pdf` e preenche os campos aplicáveis ao projeto
DeepCache. A página 4 do PDF está em branco.

---

## Cabeçalho

- **Estudantes:** [Nome Completo dos Estudantes]
- **Data:** [DD/MM/AAAA]
- **Link do Código:** [GitHub / Google Colab]

---

# 1. Artigo e Código Público Escolhido

## Informações sobre o artigo escolhido

- **Título:** DeepCache: Accelerating Diffusion Models for Free
- **Autores:** Xinyin Ma, Gongfan Fang e Xinchao Wang
- **Ano/Conferência:** CVPR 2024
- **Link arXiv/DOI:** https://openaccess.thecvf.com/content/CVPR2024/html/Ma_DeepCache_Accelerating_Diffusion_Models_for_Free_CVPR_2024_paper.html
- **Código original:** https://github.com/horseee/DeepCache

## Tipo de modelo proposto

- [ ] VAE
- [ ] GAN
- [X] Difusão
- [ ] Outro: [Nome]

## Objetivo do artigo em 2 linhas

DeepCache resolve o alto custo computacional da inferência iterativa da U-Net em modelos de difusão, cacheando e reutilizando features de alta resolução que mudam lentamente entre passos consecutivos de remoção de ruído, sem retreinamento.  
O artigo avalia Stable Diffusion em prompts do PartiPrompts e MS-COCO 2017, além de modelos LDM no ImageNet; o projeto desta entrega usa geração Text-to-Image com Stable Diffusion 1.5.

---

# 2. Execução da Base Pública

## Ambiente de Execução

- **Plataforma:** [X] Google Colab
- **Acelerador / VRAM:** NVIDIA Tesla T4, 15 GB
- **Framework e bibliotecas importantes:** Python 3, PyTorch 2.x, Diffusers, Transformers, Accelerate, DeepCache, LPIPS e TorchMetrics

## Uso do código original

- [ ] Código executou sem problemas.
- [X] Código precisou de ajustes de dependências/caminhos.
- [ ] Não executou por erro crítico.

**Ajustes previstos:**

1. Instalar o pacote `DeepCache` no ambiente do Colab.
2. Instalar uma versão do Diffusers compatível com o código do helper oficial.
3. Carregar `runwayml/stable-diffusion-v1-5` em `torch.float16`.
4. Utilizar o scheduler DDIM e fixar o gerador aleatório em `seed=42`.
5. Aplicar slicing de atenção e VAE para manter a geração dentro dos 15 GB da T4.
6. Medir tempo somente em volta da chamada `pipe(...)`, com `torch.cuda.synchronize()`.

**Execução da base pública:**

```python
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from DeepCache import DeepCacheSDHelper

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(cache_interval=3, cache_branch_id=0)
helper.enable()
images = pipe(
    prompt,
    num_inference_steps=50,
    generator=torch.Generator("cuda").manual_seed(42),
).images
helper.disable()
```

No modo baseline, o helper permanece desativado. No modo DeepCache oficial, `cache_interval=3` atualiza o cache periodicamente e `cache_branch_id=0` seleciona o ramo inicial da U-Net. Todo prompt será processado nas três versões com os mesmos passos, resoluções, seed e escala de orientação.

## Uso de pesos

- [X] Checkpoint pré-treinado oficial: `runwayml/stable-diffusion-v1-5`
- [ ] Treinado do zero: [X épocas]

---

# 3. Modificação Implementada na Base Pública

## Hipótese da modificação

A substituição do intervalo fixo $N$ por um gatilho adaptativo pode reduzir ou aumentar a frequência de atualização da U-Net conforme a mudança real do latent. A hipótese é que latents que variam pouco permitam reutilizar o cache por mais passos, mantendo qualidade próxima ao baseline, enquanto mudanças maiores provocam atualização completa do ramo cacheado antes de haver perda acumulada.

## O que foi alterado

- O comportamento original `step % cache_interval == 0` foi substituído por uma decisão adaptativa.
- Em cada etapa, calcula-se a distância relativa L2:

$$
d_t =
\frac{\left\|z_t-z_{t_{\text{last}}}\right\|_2}
{\left\|z_{t_{\text{last}}}\right\|_2+\epsilon}
$$

- O cache é atualizado somente quando $d_t>\tau$.
- `cache_branch_id=0` continua fixo, assim a intervenção modifica somente a política de atualização do cache.
- Não se alteram os pesos, o scheduler DDIM, o sampler, o prompt, a resolução ou o número de passos.

## Motivação teórica

O intervalo fixo presume que toda imagem percorre o processo de denoising com a mesma velocidade de mudança. Essa suposição pode ser conservadora ou agressiva conforme a etapa e o prompt. Medir a mudança do latent aproxima o gatilho da dinâmica real da geração e pode evitar atualizações redundantes em regiões lentas ou atrasar o cache quando a representação varia rapidamente.

## Resultado esperado

- Espera-se reduzir parte da latência do DeepCache oficial ou manter um speedup comparável.
- Espera-se qualidade visual próxima ao SD sem cache e ao DeepCache oficial.
- Espera-se manter ou reduzir o pico de VRAM em relação ao baseline.
- O teste é considerado positivo se houver speedup mensurável sem perda relevante de CLIP Score e LPIPS.

## Trecho de Código Modificado — Code Diff

**Arquivo alterado:** `notebooks/projeto_deepcache_colab.ipynb`

### Código original (BEFORE)

```python
def should_update_cache(step):
    return step % cache_interval == 0
```

### Código modificado (AFTER)

```python
relative_l2 = torch.linalg.vector_norm(
    z_t.float() - z_last.float()
) / (torch.linalg.vector_norm(z_last.float()) + 1e-8)

should_update_cache = relative_l2.item() > tau
```

## Explicação da mudança no código em uma linha

O hook substitui a periodicidade fixa por atualização completa da U-Net somente quando a distância relativa L2 entre latents consecutivos ultrapassa o limiar $\tau$.

---

# 4. Resultados Obtidos (Caso de Sucesso)

## Comparativo Visual

| Prompt | Sem Cache — Stable Diffusion | DeepCache Oficial | DeepCache Adaptativo |
|---|---|---|---|
| 1 | [Inserir imagem] | [Inserir imagem] | [Inserir imagem] |
| 2 | [Inserir imagem] | [Inserir imagem] | [Inserir imagem] |
| 3 | [Inserir imagem] | [Inserir imagem] | [Inserir imagem] |
| 4 | [Inserir imagem] | [Inserir imagem] | [Inserir imagem] |
| 5 | [Inserir imagem] | [Inserir imagem] | [Inserir imagem] |
| 6 | [Inserir imagem] | [Inserir imagem] | [Inserir imagem] |
| 7 | [Inserir imagem] | [Inserir imagem] | [Inserir imagem] |
| 8 | [Inserir imagem] | [Inserir imagem] | [Inserir imagem] |
| 9 | [Inserir imagem] | [Inserir imagem] | [Inserir imagem] |
| 10 | [Inserir imagem] | [Inserir imagem] | [Inserir imagem] |

**Arquivo gerado pelo notebook:** `deepcache-experiment/results/comparacao_visual.png`

## Análise Visual

[Comparar nitidez, textura, consistência com o prompt, variedade de composição, ruído e artefatos entre Stable Diffusion sem cache, DeepCache oficial e DeepCache adaptativo. Mencionar apenas diferenças observadas nas imagens geradas.]

## Métricas e Desempenho

| Método | Latência média por imagem (s) | Speedup | Pico de VRAM (MB) | CLIP Score médio | LPIPS médio versus sem cache |
|---|---:|---:|---:|---:|---:|
| Stable Diffusion sem cache | [_____] | 1.00x | [_____] | [_____] | 0.000 |
| DeepCache oficial | [_____] | [_____]x | [_____] | [_____] | [_____] |
| DeepCache adaptativo | [_____] | [_____]x | [_____] | [_____] | [_____] |

### Métricas por prompt

| Prompt | Latência sem cache (s) | Latência oficial (s) | Latência adaptativa (s) | Speedup oficial | Speedup adaptativo | LPIPS oficial | LPIPS adaptativo | CLIP oficial | CLIP adaptativo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] |
| 2 | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] |
| 3 | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] |
| 4 | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] |
| 5 | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] |
| 6 | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] |
| 7 | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] |
| 8 | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] |
| 9 | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] |
| 10 | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] | [_____] |

### A hipótese se confirmou?

- [ ] Sim
- [ ] Parcialmente
- [ ] Não

**Justificativa curta:** [Preencher com os valores reais do notebook.]

---

# 5. Diagnóstico de Falhas e Post-Mortem (Caso NÃO Tenha Gerado Resultados)

## Tipo de Falha Encontrada

- [ ] Código legado com dependências incompatíveis / links quebrados
- [ ] Limitação de GPU / Memória insuficiente (CUDA Out of Memory)
- [ ] Instabilidade numérica (Loss virou NaN / Inf)
- [ ] Colapso do modelo generativo (ruído puro ou imagem inválida)
- [ ] Outro: [Descreva]

## Log de Erro ou Evidência Visual da Falha

[Cole aqui o stack trace, mensagem da T4 ou imagem que evidencie a falha.]

## Causa-Raiz Técnica

[Identifique se a falha foi causada por infraestrutura, incompatibilidade de versões, VRAM ou erro do modelo.]

## O que vocês tentaram para contornar?

- [ ] Reduziu batch size / resolução
- [ ] Alterou learning rate / otimizador
- [ ] Simplificou camadas / removeu blocos
- [ ] Reduziu os 50 passos DDIM
- [ ] Desativou safety checker ou ativou slicing
- [ ] Outro: [Descreva]

**Resultado das tentativas:** [Comente os resultados.]

**O que seria necessário para viabilizar o experimento:** [Ex.: GPU com maior VRAM, compatibilidade fixa de versões, tempo de execução maior ou menor resolução.]

---

# 6. Conclusão Rápida

**Principal aprendizado prático:** [Em 1 ou 2 frases, explicar o que a reprodução e a intervenção adaptativa demonstraram sobre aceleração de geração Text-to-Image, qualidade visual e uso da T4.]

---

## Inventário das 4 páginas do PDF

1. **Página 1:** cabeçalho, seção 1 e início da seção 2.
2. **Página 2:** final da seção 2 e seções 3 e início da seção 4.
3. **Página 3:** final da seção 4, seção 5 e seção 6.
4. **Página 4:** página em branco.
