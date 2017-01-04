from setuptools import setup

setup(name='smith_agent_model',
      version='0.1',
      description="Eliot R. Smith's agent based model in Python",
      url='http://github.com/kubikb/smith_agent_model',
      author='Balint Kubik',
      author_email='kubikbalint@gmail.com',
      license='MIT',
      packages=['smith_agent_model'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[
          'numpy',
      ])