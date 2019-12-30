#include <FeatureCluster.h>

namespace ark {

    Vocabulary::Vocabulary() {
        ;
    }

    bool Vocabulary::addDescriptor(const cv::Mat &des) {

        descriptors.push_back(des);
    }

    bool Vocabulary::createVocabulary() {

        vocab.create(descriptors);
    }

    bool Vocabulary::calculateBowVec() {

        for (const auto &des : descriptors) {

            DBoW3::BowVector bw;
            vocab.transform(des, bw);
            bowVector.push_back(bw);
        }

    }

    int Vocabulary::calculateNN(const cv::Mat &des) {

        DBoW3::BowVector v2;
        vocab.transform(des, v2);

        int cnt = 0;
        int maxPos = 0;
        float maxScore = -1.0;
        float Score = 0.f;
        for (const auto &v1: bowVector) {

            Score = vocab.score(v1, v2);
            if (maxScore < Score) {

                maxScore = Score;
                maxPos = cnt;
            }
            cnt++;
        }

        return maxPos;
    }
}
