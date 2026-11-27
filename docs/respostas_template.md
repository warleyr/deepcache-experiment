# Respostas do Template: Relatório Projeto, Modelos Generativos para Imagens

Este documento preenche os campos do `template-relatorio-projeto.pdf` para o projeto DeepCache. O trabalho foi desenvolvido individualmente, com um único autor. Os marcadores `[PREENCHER APÓS EXECUÇÃO NO COLAB]` indicam os campos que dependem da execução do notebook `notebooks/projeto_deepcache_colab.ipynb` na GPU T4.

---

## Cabeçalho

- **Estudante (projeto individual):** [Nome Completo]
- **Data:** [PREENCHER APÓS EXECUÇÃO NO COLAB]
- **Link do Código:** [URL do repositório GitHub] / [URL do notebook no Google Colab]

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

DeepCache reduz o custo da inferência iterativa da U-Net em modelos de difusão. Ele reaproveita features de alta resolução que mudam pouco entre passos consecutivos de remoção de ruído, sem retreinar o modelo.
O artigo avalia Stable Diffusion com prompts do PartiPrompts e do MS-COCO 2017, além de LDM no ImageNet. Este projeto usa Text-to-Image com Stable Diffusion 1.5 e 5 prompts oficiais.

---

# 2. Execução da Base Pública

## Ambiente de Execução

- **Plataforma:** [X] Google Colab
- **Acelerador / VRAM:** NVIDIA Tesla T4, 15 GB
- **Framework e bibliotecas:** Python 3.x, PyTorch (versão do runtime do Colab), Diffusers 0.43.x, Transformers 5.x, DeepCache 0.1.1, LPIPS e TorchMetrics. As versões exatas são impressas pelo Bloco 1 do notebook.

## Uso do código original

- [ ] Código executou sem problemas.
- [X] Código precisou de ajustes de dependências/caminhos: o notebook embute os 5 prompts oficiais e valida o scheduler, sem depender de clonar o repositório.
- [ ] Não executou por erro crítico.

**Configuração do experimento:**

- Modelo: `runwayml/stable-diffusion-v1-5`, float16, atenção fatiada (`attention slicing = max`).
- Scheduler: PLMS, implementado como `PNDMScheduler(skip_prk_steps=True)` com 50 passos. Antes de cada medição, o notebook valida que não há timesteps duplicados. Se houver, troca para `LMSDiscreteScheduler`, que é o fallback. [PREENCHER APÓS EXECUÇÃO NO COLAB: scheduler usado, impresso pelo Bloco 3]
- Guidance 7.5, resolução 512×512, seed 42, mesmos passos para as três versões.
- DeepCache oficial: `DeepCacheSDHelper`, `cache_interval=3`, `cache_branch_id=0`.

**Execução do baseline oficial:**

```python
from DeepCache import DeepCacheSDHelper

helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(cache_interval=3, cache_branch_id=0)

helper.enable()
try:
    image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=7.5,
                 generator=torch.Generator('cuda').manual_seed(42)).images[0]
finally:
    helper.disable()
```

Sem cache, o helper permanece desativado. Com `cache_interval=3`, a U-Net completa é calculada nos passos cujo índice é múltiplo de 3 (17 passos de 50). Nos demais, os blocos reutilizam a saída do passo anterior.

## Uso de pesos

- [X] Checkpoint pré-treinado oficial: `runwayml/stable-diffusion-v1-5`
- [ ] Treinado do zero: não aplicável. Nenhum treinamento foi feito.

---

# 3. Modificação Implementada na Base Pública

## Hipótese da modificação

O intervalo fixo `cache_interval` atualiza o cache no mesmo ritmo para todos os prompts. A hipótese é que o gatilho adaptativo, baseado na mudança real do latent, reaproveita o cache por mais passos quando o latent muda pouco e atualiza com mais frequência quando muda muito. Assim, a qualidade fica próxima do DeepCache oficial e a economia de passos completos aumenta.

## O que foi alterado

- A decisão de atualização do cache do `DeepCacheSDHelper` foi substituída. O método `is_skip_step` original usa um intervalo fixo. A versão adaptativa força o cálculo completo quando `refresh=True` e delega ao método original nos demais passos.
- Em cada fim de passo, um callback calcula a distância relativa L2 entre o latent atual e o último latent recalculado, e define `refresh` para a próxima chamada da U-Net.
- Não se alteram os pesos, o scheduler, o sampler, os prompts, a resolução, o número de passos nem a seed.

## Motivação teórica

O artigo mostra que as features de alta resolução da U-Net mudam pouco entre passos consecutivos. Essa mudança depende do prompt e da etapa do denoising. Medir a variação do latent aproxima o gatilho da dinâmica real da geração, em vez de assumir uma periodicidade fixa.

## Resultado esperado

- Economia de passos completos da U-Net em relação ao DeepCache oficial, com latência menor ou comparável.
- Qualidade próxima ao Stable Diffusion sem cache, medida por CLIP Score e LPIPS.
- Pico de VRAM igual ou menor que o do baseline.
- O teste é positivo se houver redução mensurável de passos ou latência, sem queda relevante de CLIP Score.

## Trecho de Código Modificado (Code Diff)

**Arquivo alterado:** `notebooks/projeto_deepcache_colab.ipynb`, Bloco 4 (`deepcache-adaptive`) e Bloco 5 (`calibracao`).

### Código original (BEFORE): `DeepCacheSDHelper.is_skip_step` no pacote oficial `DeepCache` 0.1.1

```python
def is_skip_step(self, block_i, layer_i, blocktype = "down"):
    self.start_timestep = self.cur_timestep if self.start_timestep is None else self.start_timestep
    cache_interval, cache_layer_id, cache_block_id, skip_mode = \
        self.params['cache_interval'], self.params['cache_layer_id'], self.params['cache_block_id'], self.params['skip_mode']
    if skip_mode == 'uniform':
        if (self.cur_timestep-self.start_timestep) % cache_interval == 0: return False
    if block_i > cache_block_id or blocktype == 'mid':
        return True
    if block_i < cache_block_id: return False
    return layer_i >= cache_layer_id if blocktype == 'down' else layer_i > cache_layer_id
```

A decisão é tomada por `cur_timestep`, o índice do timestep no scheduler. O método é chamado por `wrap_block_forward` em cada bloco e em cada passo.

### Código modificado (AFTER): subclasse adaptativa e callback

```python
class AdaptiveDeepCacheSDHelper(DeepCacheSDHelper):
    def __init__(self, pipe=None, tau=None):
        super().__init__(pipe=pipe)
        self.tau = tau

    def reset_states(self):
        super().reset_states()
        self.refresh = True
        self.ref_latent = None
        self.distances = []
        self.called_steps = set()
        self.full_steps = set()

    def is_skip_step(self, block_i, layer_i, blocktype='down'):
        if self.tau is None:
            skip = super().is_skip_step(block_i, layer_i, blocktype)
        elif self.refresh:
            skip = False
        else:
            skip = super().is_skip_step(block_i, layer_i, blocktype)

        if block_i == 0 and layer_i == 0 and blocktype == 'down':
            self.called_steps.add(self.cur_timestep)
            if not skip:
                self.full_steps.add(self.cur_timestep)
        return skip


def make_adaptive_callback(helper):
    def _on_step_end(pipe, step, timestep, callback_kwargs):
        z = callback_kwargs['latents'].detach().float()
        if helper.ref_latent is None:
            distance = float('inf')
        else:
            distance = (
                torch.linalg.vector_norm(z - helper.ref_latent)
                / (torch.linalg.vector_norm(helper.ref_latent) + 1e-8)
            ).item()
        helper.distances.append(distance)
        helper.refresh = distance > helper.tau
        if helper.refresh:
            helper.ref_latent = z.clone()
        return callback_kwargs
    return _on_step_end
```

Chamada: `kwargs['callback_on_step_end'] = make_adaptive_callback(helper)` e `kwargs['callback_on_step_end_tensor_inputs'] = ['latents']`.

**Explicação da mudança no código (1 linha):** a U-Net só é recalculada por inteiro quando a distância relativa L2 entre latents supera `tau`, e a decisão vale para o passo seguinte.

**Limitação conhecida:** o primeiro passo não tem latent de referência, então o passo 1 também é recalculado por inteiro. A contagem de passos completos inclui esse passo.

## Escolha e calibração do limiar tau

- **Candidatos:** tau ∈ {0,015; 0,020; 0,030}.
- **Prompts de calibração (exclusivos):** 3 prompts que não aparecem na avaliação final, listados no Bloco 2 do notebook.
- **Critério:** entre os candidatos com queda de CLIP Score de no máximo 0,01 em relação ao Stable Diffusion sem cache nos prompts de calibração, escolhe-se o que economiza mais passos completos. Se nenhum atender, escolhe-se o de menor queda.
- **Avaliação final:** o tau escolhido roda nos 5 prompts oficiais de `data/prompts_geracao.txt`, que estão embutidos no notebook com `assert len(OFFICIAL_PROMPTS) == 5`.
- **Tabela de calibração:** [PREENCHER APÓS EXECUÇÃO NO COLAB: copiar a tabela impressa pelo Bloco 5]
- **tau escolhido:** [PREENCHER APÓS EXECUÇÃO NO COLAB: valor impresso pelo Bloco 5]

---

# 4. Resultados Obtidos (Caso de Sucesso)

## Comparativo Visual

| Prompt | Sem Cache (Stable Diffusion) | DeepCache Oficial | DeepCache Adaptativo |
|---|---|---|---|
| 1 | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] |
| 2 | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] |
| 3 | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] |
| 4 | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] |
| 5 | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] |

**Arquivo gerado pelo notebook:** `results/comparacao_visual.png` (Bloco 7). Para inserir as imagens, use os arquivos `results/prompt_XX_sem_cache.png`, `results/prompt_XX_deepcache_oficial.png` e `results/prompt_XX_deepcache_adaptativo.png`.

## Análise Visual

[PREENCHER APÓS EXECUÇÃO NO COLAB: comparar nitidez, textura, aderência ao prompt, ruído e artefatos entre as três versões. Descrever só o que aparece nas imagens geradas.]

## Métricas e Desempenho

**Onde colar:** copiar a tabela impressa pelo Bloco 7 do notebook (também salva em `results/tabela_resultados.md`) no lugar da tabela abaixo.

| Método | Latência média (s) | Speedup | Pico de VRAM (MB) | Passos completos executados (de 50) | CLIP Score médio | LPIPS médio vs. sem cache |
|---|---:|---:|---:|---:|---:|---:|
| Stable Diffusion sem cache | [PREENCHER APÓS EXECUÇÃO NO COLAB] | 1,00x | [PREENCHER APÓS EXECUÇÃO NO COLAB] | 50 | [PREENCHER APÓS EXECUÇÃO NO COLAB] | 0,0000 |
| DeepCache oficial | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] |
| DeepCache adaptativo (tau escolhido) | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] | [PREENCHER APÓS EXECUÇÃO NO COLAB] |

**Detalhes por prompt:** `results/metricas_resultados.csv`. **Resumo em JSON, com calibração:** `results/resumo_resultados.json`.

### A hipótese se confirmou?

- [ ] Sim
- [ ] Parcialmente
- [ ] Não

**Justificativa curta:** [PREENCHER APÓS EXECUÇÃO NO COLAB: citar o número de passos completos economizados, a variação de latência e a queda de CLIP Score e LPIPS, com os valores reais do notebook.]

---

# 5. Diagnóstico de Falhas e Post-Mortem

Preencher esta seção somente se a execução falhar. Se o notebook concluir, marcar como não aplicável.

## Tipo de Falha Encontrada

- [ ] Código legado com dependências incompatíveis / links quebrados
- [ ] Limitação de GPU / Memória insuficiente (CUDA Out of Memory)
- [ ] Instabilidade numérica (Loss virou NaN / Inf)
- [ ] Colapso do modelo generativo (ruído puro ou imagem inválida)
- [ ] Outro: [PREENCHER APÓS EXECUÇÃO NO COLAB, se houver]

## Log de Erro ou Evidência Visual da Falha

[PREENCHER APÓS EXECUÇÃO NO COLAB, se houver falha.]

## Causa-Raiz Técnica

[PREENCHER APÓS EXECUÇÃO NO COLAB, se houver falha.]

## O que foi tentado para contornar

- [ ] Reduziu batch size / resolução
- [ ] Alterou o scheduler (PNDM para LMS)
- [ ] Reduziu os passos de inferência
- [ ] Outro: [PREENCHER APÓS EXECUÇÃO NO COLAB, se houver]

**Resultado das tentativas:** [PREENCHER APÓS EXECUÇÃO NO COLAB, se houver falha.]

---

# 6. Conclusão Rápida

**Principal aprendizado prático:** [PREENCHER APÓS EXECUÇÃO NO COLAB: uma ou duas frases sobre o que a reprodução e a intervenção adaptativa mostraram sobre aceleração de Text-to-Image, qualidade visual e uso da T4.]
