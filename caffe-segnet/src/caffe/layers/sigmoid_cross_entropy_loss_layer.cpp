#include <algorithm>
#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (this->layer_param_.loss_param().has_normalization()) {
    normalization_ = this->layer_param_.loss_param().normalization();
  } else if (this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

// TODO(shelhamer) loss normalization should be pulled up into LossLayer,
// instead of duplicated here and in SoftMaxWithLossLayer
template <typename Dtype>
Dtype SigmoidCrossEntropyLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  Dtype temp_loss_pos = 0;
  Dtype temp_loss_neg = 0;
  Dtype count_pos = 0;
  Dtype count_neg = 0;
  //Dtype summ = 0;
  //Dtype lambda = 100;

  lambda_ = this->layer_param_.loss_param().lambda();

  const int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();

  int valid_count = 0;
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
      temp_loss_pos = 0;
      temp_loss_neg = 0;
      count_pos = 0;
      count_neg = 0;
      for (int j = 0; j < dim; j ++) {
	const int target_value = static_cast<int>(target[i*dim+j]);
	if (has_ignore_label_ && target_value == ignore_label_) {
	      continue;
	    }
         if (target[i*dim+j] > 0.4) {
        	count_pos ++;
        	temp_loss_pos -= input_data[i*dim + j] * (target[i*dim+j] - (input_data[i*dim + j] >= 0)) -
                	log(1 + exp(input_data[i*dim + j] - 2 * input_data[i*dim + j] * (input_data[i*dim + j] >= 0)));
    	}
    	else if (target[i*dim+j] < 0.4) {
        	count_neg ++;
        	temp_loss_neg -= input_data[i*dim + j] * (target[i*dim+j] - (input_data[i*dim + j] >= 0)) -
                	log(1 + exp(input_data[i*dim + j] - 2 * input_data[i*dim + j] * (input_data[i*dim + j] >= 0)));
    	}
	++valid_count;
     } 
     loss_pos += temp_loss_pos * lambda_ * count_neg / (count_pos + count_neg);
     loss_neg += temp_loss_neg * count_pos / (count_pos + count_neg);
  }

  //summ = 0;
  //summ = count_pos + count_neg;

  //LOG(INFO) << "valid_count: " << valid_count;
  //LOG(INFO) << "count_pos: " << count_pos << "count_neg: " << "," << count_neg;
  //LOG(INFO) << "summ: " << summ;
  //LOG(INFO) << "num: " << num;
  //LOG(INFO) << "dim: " << dim;
  //LOG(INFO) << "loss_pos: " << loss_pos;
  //LOG(INFO) << "loss_neg: " << loss_neg;

  normalizer_ = get_normalizer(normalization_, valid_count);
  loss = (loss_pos * 1 + loss_neg) / num;
  top[0]->mutable_cpu_data()[0] = loss / normalizer_; 
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    Dtype count_pos = 0;
    Dtype count_neg = 0;
    int dim = bottom[0]->count() / bottom[0]->num();
    const int num = bottom[0]->num();

    lambda_ = this->layer_param_.loss_param().lambda();

    for (int i = 0; i < num; ++i) {
    	count_pos = 0;
    	count_neg = 0;
    	for (int j = 0; j < dim; j ++) {
           	if (target[i*dim+j]  > 0.4) {
                	count_pos ++;
        	}
        	else if (target[i*dim+j] < 0.4) {
                	count_neg ++;
        	}
     	}
    	for (int j = 0; j < dim; j ++) {
		if (has_ignore_label_) {
			const int target_value = static_cast<int>(target[i*dim+j]);
			if (target_value == ignore_label_) {
         		 bottom_diff[i * dim + j] = 0;
			}
		}
        	if (target[i*dim+j] > 0.4) {
               		bottom_diff[i * dim + j] *= lambda_ * count_neg / (count_pos + count_neg);
        	}
        	else if (target[i*dim+j] < 0.4) {
                	bottom_diff[i * dim + j] *= count_pos / (count_pos + count_neg);
        	}
     	}
    }
    const Dtype loss_weight = top [0]->cpu_diff()[0] / normalizer_;
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
