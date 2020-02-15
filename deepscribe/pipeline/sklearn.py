import luigi
from deepscribe.pipeline.selection import SelectDatasetTask
import numpy as np
import sklearn as sk
import sklearn.metrics
import sklearn.linear_model
import sklearn.neighbors
import sklearn.ensemble
import pickle as pk


class TrainSKLModelFromDefinitionTask(luigi.Task):
    imgfolder = luigi.Parameter()
    hdffolder = luigi.Parameter()
    modelsfolder = luigi.Parameter()
    target_size = luigi.IntParameter()  # standardizing to square images
    keep_categories = luigi.ListParameter()
    fractions = luigi.ListParameter()  # train/valid/test fraction
    model_definition = luigi.Parameter()  # JSON file with model definition specs

    def requires(self):
        return SelectDatasetTask(
            self.imgfolder,
            self.hdffolder,
            self.target_size,
            self.keep_categories,
            self.fractions,
        )

    def get_model(self):
        raise NotImplementedError

    def run(self):

        self.output().makedirs()

        data = np.load(self.input().path)

        input_x = data["train_imgs"]

        # reshape data for input to sklearn classifier

        n_examples = input_x.shape[0]

        vector_dim = input_x.shape[1] * input_x.shape[2] * input_x.shape[3]

        input_x_flattened = input_x.reshape(n_examples, vector_dim)
        # TODO: set terms from input file
        model = self.get_model()

        model.fit(input_x_flattened, data["train_labels"])

        # compute performance on validation data

        # reshape

        valid_x = data["valid_imgs"]

        valid_x_flattened = valid_x.reshape(
            valid_x.shape[0], valid_x.shape[1] * valid_x.shape[2] * valid_x.shape[3]
        )

        pred_valid_y = model.predict(valid_x_flattened)

        acc = sk.metrics.accuracy_score(data["valid_labels"], pred_valid_y)

        print("Model accuracy on validation data: {}".format(acc))

        balanced_acc = sk.metrics.balanced_accuracy_score(
            data["valid_labels"], pred_valid_y
        )
        print("balanced accuracy score on validation data: {}".format(balanced_acc))

        # compute AUC score

        # convert data to categorical

        validation_onehot = kr.utils.to_categorical(data["valid_labels"])

        auc_macro = sk.metrics.roc_auc_score(
            validation_onehot, model.predict_proba(valid_x_flattened), average="macro"
        )

        print("Macro AUC on validation data: {}".format(auc_macro))

        auc_micro = sk.metrics.roc_auc_score(
            validation_onehot, model.predict_proba(valid_x_flattened), average="micro"
        )

        print("Micro AUC on validation data: {}".format(auc_micro))

        f1_macro = sk.metrics.f1_score(
            data["valid_labels"], pred_valid_y, average="macro"
        )

        print("Macro F1 on validation data: {}".format(f1_macro))

        f1_micro = sk.metrics.f1_score(
            data["valid_labels"], pred_valid_y, average="micro"
        )

        print("Micro F1 on validation data: {}".format(f1_micro))

        # builtin luigi doesn't work with bytes mode?
        with open(self.output().path, "wb") as outf:
            pk.dump(model, outf)

    def output(self):
        p = Path(self.model_definition)
        p_data = Path(self.input().path)

        return luigi.LocalTarget(
            "{}/{}_{}/trained.pkl".format(self.modelsfolder, p.stem, p_data.stem)
        )


# trains a linear model from sklearn
class TrainLinearModelTask(TrainSKLModelFromDefinitionTask):
    def get_model(self):
        return sk.linear_model.LogisticRegression(verbose=True, n_jobs=-1)


# TODO: refactor to avoid repeated code
class TrainKNNModelTask(TrainSKLModelFromDefinitionTask):
    def get_model(self):
        return sk.neighbors.KNeighborsClassifier(n_jobs=-1)


class TrainRFModelTask(TrainSKLModelFromDefinitionTask):
    def get_model(self):

        # load model
        with open(self.model_definition, "r") as modelf:
            model_params = json.load(modelf)

        return sk.ensemble.RandomForestClassifier(
            verbose=1, n_jobs=-1, n_estimators=model_params["estimators"]
        )


class TrainGBModelTask(TrainSKLModelFromDefinitionTask):
    def get_model(self):
        # load model
        with open(self.model_definition, "r") as modelf:
            model_params = json.load(modelf)

        return sk.ensemble.GradientBoostingClassifier(
            verbose=1, n_estimators=model_params["estimators"]
        )
