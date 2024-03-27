import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { type BaseLLMParams } from "@langchain/core/language_models/llms";
import { LLMResult } from "@langchain/core/outputs";
import { OpenAI } from "@langchain/openai";

/**
 * Interface that extends the LLMInput interface and adds additional
 * parameters specific to the VLLMOpenAI class.
 */
export interface VLLMOpenAIInput extends BaseLLMParams {
  /**
   * The name or path of a HuggingFace Transformers model.
   */
  model: string;

  /**
   * The number of GPUs to use for distributed execution with tensor parallelism.
   */
  tensorParallelSize?: number;

  /**
   * Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.
   */
  trustRemoteCode?: boolean;

  /**
   * Number of output sequences to return for the given prompt.
   */
  n?: number;

  /**
   * Number of output sequences that are generated from the prompt.
   */
  bestOf?: number;

  /**
   * Float that penalizes new tokens based on whether they appear in the generated text so far
   */
  presencePenalty?: number;

  /**
   * Float that penalizes new tokens based on their frequency in the generated text so far
   */
  frequencyPenalty?: number;

  /**
   * Float that controls the randomness of the sampling.
   */
  temperature?: number;

  /**
   * Float that controls the cumulative probability of the top tokens to consider.
   */
  topP?: number;

  /**
   * Integer that controls the number of top tokens to consider.
   */
  topK?: number;

  /**
   * Whether to use beam search instead of sampling.
   */
  useBeamSearch?: boolean;

  /**
   * List of strings that stop the generation when they are generated.
   */
  stop?: string[];

  /**
   * Whether to ignore the end of sequence token.
   */
  ignoreEos?: boolean;

  /**
   * Maximum number of tokens to generate per output sequence.
   */
  maxNewTokens?: number;

  /**
   * Number of log probabilities to return per output token.
   */
  topLogprobs?: number;

  /**
   * Whether to return log probabilities.
   */
  logprobs?: boolean;

  /**
   * The data type for the model weights and activations.
   */
  dtype?: string;

  /**
   * Directory to download and load the weights. (Default to the default cache dir of huggingface)
   */
  downloadDir?: string;

  /**
   * Holds any model parameters valid for `vllm.LLM` call not explicitly specified.
   */
  vllmKwargs?: Record<string, unknown>;

  openAIApiKey?: string;
  openAIApiBase?: string;
}

/**
 * Class that represents the VLLMOpenAI language model. It extends the
 * BaseOpenAI class and implements the VLLMOpenAIInput interface.
 */
export class VLLMOpenAI extends OpenAI implements VLLMOpenAIInput {
  static lc_name() {
    return "VLLMOpenAI";
  }

  static lc_description() {
    return "vLLM OpenAI-compatible API client";
  }

  static lc_serializable() {
    return false;
  }

  model = "";

  tensorParallelSize?: number;

  trustRemoteCode?: boolean;

  n = 1;

  bestOf?: number;

  presencePenalty = 0;

  frequencyPenalty = 0;

  temperature = 1;

  topP = 1;

  topK?: number;

  useBeamSearch?: boolean;

  stop?: string[];

  ignoreEos?: boolean;

  maxNewTokens?: number;

  logprobs?: boolean;

  topLogprobs?: number;

  dtype?: string;

  downloadDir?: string;

  vllmKwargs?: Record<string, unknown>;

  constructor(fields: VLLMOpenAIInput) {
    super(fields);
    this.model = fields.model;
    this.tensorParallelSize = fields.tensorParallelSize;
    this.trustRemoteCode = fields.trustRemoteCode;
    this.n = fields.n ?? this.n;
    this.bestOf = fields.bestOf;
    this.presencePenalty = fields.presencePenalty ?? this.presencePenalty;
    this.frequencyPenalty = fields.frequencyPenalty ?? this.frequencyPenalty;
    this.temperature = fields.temperature ?? this.temperature;
    this.topP = fields.topP ?? this.topP;
    this.topK = fields.topK;
    this.useBeamSearch = fields.useBeamSearch;
    this.stop = fields.stop;
    this.ignoreEos = fields.ignoreEos;
    this.maxNewTokens = fields.maxNewTokens;
    this.logprobs = fields.logprobs;
    this.topLogprobs = fields.topLogprobs;
    this.dtype = fields.dtype;
    this.downloadDir = fields.downloadDir;
    this.vllmKwargs = fields.vllmKwargs ?? {};
    this.openAIApiKey = fields.openAIApiKey ?? '';
    this.openAIApiBase = fields.openAIApiBase ?? '';
  }

  get _invocation_params() {
    const params: Record<string, unknown> = {
      model: this.model,
      ...this._defaultParams,
      logit_bias: null,
    };

    params.openAIApiKey = this.openAIApiKey;
    params.api_base = this.openAIApiBase;

    return params;
  }

  _llmType() {
    return "vllm-openai";
  }

  _defaultParams() {
    return {
      n: this.n,
      bestOf: this.bestOf,
      maxTokens: this.maxNewTokens,
      topK: this.topK,
      topP: this.topP,
      temperature: this.temperature,
      presencePenalty: this.presencePenalty,
      frequencyPenalty: this.frequencyPenalty,
      stop: this.stop,
      ignoreEos: this.ignoreEos,
      useBeamSearch: this.useBeamSearch,
      logprobs: this.logprobs,
      topLogprobs: this.topLogprobs,
      openAIApiKey: this.openAIApiKey,
      openAIApiBase: this.openAIApiBase
    };
  }

  async _call(
    prompts: string[],
    stop?: string[],
    runManager?: CallbackManagerForLLMRun,
    kwargs?: Record<string, unknown>
  ): Promise<LLMResult> {
    console.log('### CALL');
    const params = {
      ...this._defaultParams(),
      ...kwargs,
      stop,
    };

    return await this._generate(prompts, params, runManager);
  }
}
